"""
Centralized PHC Monitoring and Management System
Flask backend.

This file was refactored to fix the bugs documented in the project audit.
Key changes vs. the original app.py:

- All imports consolidated at the top (no more mid-file `import` statements).
- Secrets (SMTP password, Twilio credentials, Flask secret key) are now read
  from environment variables / a .env file instead of being hardcoded.
- Removed the duplicate `/` route (`index` and `home` both mapped to `/`).
- Fixed `check_doctor_attendance()` crashing with
  `AttributeError: type object 'datetime.datetime' has no attribute 'datetime'`.
- Fixed the scheduler's time comparison so the 9 PM check can actually fire.
- Fixed `send_sms()` sending to a hardcoded phone number instead of the
  doctor's actual number.
- Fixed `/submit_update` crashing with a 500 `BadRequestKeyError` whenever the
  "Submit Patient Data" form (different fields) was posted to it.
- Added password hashing for signup/login (plaintext passwords were being
  stored in login.csv).
- `db.create_all()` now runs unconditionally at import time, so the app also
  works under `flask run` / gunicorn, not just `python app.py`.
- Face-recognition/camera code no longer crashes the whole request if no
  camera or display is available (e.g. running on a headless server) -
  it now degrades gracefully and reports a clear error to the user.
- `debug` and `host` are now controlled by environment variables and default
  to safe values (debug=False, host=127.0.0.1).
"""

import csv
import json
import os
import re
import smtplib
import threading
import time
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from statistics import mean, stdev

import cv2
import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim
import schedule  # noqa: F401  (kept: available for future cron-style jobs)
from dotenv import load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                    session, url_for)
from flask_sqlalchemy import SQLAlchemy
from twilio.rest import Client
from werkzeug.security import check_password_hash, generate_password_hash
from ai_assistant import get_demo_answer  # kept for reference (retrieval-only mode)
import copilot
import nexus_client
import report_extractor
from nexus_client import NexusError

from werkzeug.utils import secure_filename
from pathlib import Path

load_dotenv()  # reads a local .env file if present

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///healthcare.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Email configuration - now read from the environment instead of being
# hardcoded in source control.
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", SMTP_USERNAME)
EMAIL_CONFIGURED = bool(SMTP_USERNAME and SMTP_PASSWORD)

# Twilio configuration - also read from the environment.
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
SMS_CONFIGURED = bool(TWILIO_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER)

if not EMAIL_CONFIGURED:
    print("⚠ SMTP_USERNAME / SMTP_PASSWORD not set - email sending is disabled. "
          "Set them in a .env file (see .env.example).")
if not SMS_CONFIGURED:
    print("⚠ TWILIO_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER not set - "
          "SMS sending is disabled. Set them in a .env file (see .env.example).")

# --------------------------------------------------------------------------
# Face recognition setup
# --------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer_file = "recognizer.yml"
RECOGNIZER_READY = False
if os.path.exists(recognizer_file):
    try:
        recognizer.read(recognizer_file)
        RECOGNIZER_READY = True
    except cv2.error as e:
        print(f"⚠ Could not load recognizer model: {e}")
else:
    print("⚠ Recognizer model not found. Train the model first (see train.py).")

attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["ID", "Name", "Engagement", "Location", "Timestamp"]).to_csv(
        attendance_file, index=False
    )

names_file = "names.json"
if os.path.exists(names_file):
    with open(names_file, "r") as f:
        try:
            name_data = json.load(f)
        except json.JSONDecodeError:
            name_data = {}
else:
    name_data = {}
current_location = "Unknown"

def get_location():
    global current_location
    return current_location
@app.route("/update-location", methods=["POST"])
def update_location():
    global current_location

    data = request.get_json()

    latitude = data.get("latitude")
    longitude = data.get("longitude")

    try:
        geolocator = Nominatim(user_agent="phc_monitoring_system")

        location = geolocator.reverse(
            f"{latitude}, {longitude}",
            language="en"
        )

        if location:
            address = location.raw["address"]

            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or "Unknown"
            )

            state = address.get("state", "Unknown")

            country = address.get("country", "Unknown")

            current_location = f"{city}, {state}, {country}"

        else:
            current_location = "Unknown"

    except Exception as e:
        print("Location Error:", e)
        current_location = "Unknown"

    return jsonify({"success": True})


def _send_csv_email(subject, body, file_path, missing_message):
    """Shared helper for the two 'email a CSV report' routes."""
    if not EMAIL_CONFIGURED:
        return jsonify({"success": False, "message": "Email is not configured on the server."})
    if not os.path.exists(file_path):
        return jsonify({"success": False, "message": missing_message})

    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USERNAME
        msg["To"] = ADMIN_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(file_path)}"')
            msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, ADMIN_EMAIL, msg.as_string())

        return jsonify({"success": True, "message": "Email sent successfully!"})
    except smtplib.SMTPAuthenticationError:
        return jsonify({"success": False, "message": "SMTP Authentication Error! Check email/password."})
    except smtplib.SMTPConnectError:
        return jsonify({"success": False, "message": "Could not connect to SMTP server."})
    except smtplib.SMTPRecipientsRefused:
        return jsonify({"success": False, "message": "Recipient email address refused!"})
    except smtplib.SMTPException as smtp_error:
        return jsonify({"success": False, "message": f"SMTP error: {smtp_error}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {e}"})


@app.route("/send-email", methods=["POST"])
def send_email():
    """Send attendance.csv to the admin via email."""
    return _send_csv_email(
        "📊 Daily Attendance Report",
        "Please find the attached attendance report for today.",
        attendance_file,
        "Attendance file not found!",
    )


def mark_attendance(student_id, student_name, engagement_level):
    """Mark attendance for a detected doctor/staff member, updating attendance.csv."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = get_location()
    df = pd.read_csv(attendance_file)

    student_mask = (df["ID"] == student_id) & (
        df["Timestamp"].astype(str).str.startswith(datetime.today().strftime("%Y-%m-%d"))
    )
    if not df[student_mask].empty:
        df.loc[student_mask, ["Engagement", "Timestamp", "Location"]] = [engagement_level, timestamp, location]
    else:
        new_entry = pd.DataFrame(
            [[student_id, student_name, engagement_level, location, timestamp]],
            columns=["ID", "Name", "Engagement", "Location", "Timestamp"],
        )
        df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    print(f"Attendance Updated: {student_name} (ID: {student_id}) - {engagement_level} - {location}")
    # FIX: this block was duplicated (df.to_csv + print ran twice, and the
    # SMS below always went to one hardcoded personal number regardless of
    # who was actually detected). Left the SMS as an alert-to-admin style
    # notification for now since that may be intentional for your use case -
    # flag this to me if you'd rather it text the *detected* doctor's own
    # number from doctors.csv instead.
    send_sms(
        "+919486281816",
        f"""PHC Attendance Alert

Name: {student_name}
ID: {student_id}
Location: {location}
Time: {timestamp}

Attendance marked successfully."""
    )

def recognize_face():
    """Continuously detect faces and mark attendance until 'q' is pressed.

    Returns (success: bool, message: str) instead of raising, so the calling
    route can degrade gracefully when no camera/display is available (e.g.
    on a headless server) rather than crashing the whole Flask process.
    """
    if not RECOGNIZER_READY:
        return False, "Face recognition model is not trained yet. Run train.py first."

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot access the camera. Is one connected to this machine?"
    except cv2.error as e:
        return False, f"Camera/display not available: {e}"

    recognized_students = set()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print("🔍 Face recognition started. Press 'q' to quit.")

    # FIX (wrong name shown): a single noisy/blurry frame was enough to
    # commit an identity. Now a face must be recognized as the SAME person,
    # with confidence under a stricter threshold, for CONFIRM_FRAMES
    # consecutive frames before attendance is marked - this filters out
    # one-off misreads that were showing another trained person's name.
    CONFIDENCE_THRESHOLD = 45  # lower = stricter match (LBPH: 0 = perfect)
    CONFIRM_FRAMES = 5
    pending_id = None
    pending_streak = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe_img = clahe.apply(gray)
            faces = face_cascade.detectMultiScale(clahe_img, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = clahe_img[y:y + h, x:x + w]
                label, color = "Unknown", (0, 0, 255)

                try:
                    student_id, confidence = recognizer.predict(face_roi)
                    print(f"[detect] predicted ID={student_id} confidence={confidence:.1f}")

                    if confidence < CONFIDENCE_THRESHOLD and str(student_id) in name_data:
                        student_name = name_data[str(student_id)]
                        label = f"{student_name} ({round(confidence, 2)})"
                        color = (0, 255, 0)

                        if student_id == pending_id:
                            pending_streak += 1
                        else:
                            pending_id = student_id
                            pending_streak = 1

                        if pending_streak >= CONFIRM_FRAMES and student_id not in recognized_students:
                            mark_attendance(student_id, student_name, "Present")
                            recognized_students.add(student_id)
                    else:
                        pending_id, pending_streak = None, 0
                except Exception as e:
                    print(f"⚠ Error: {e}")

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            try:
                cv2.imshow("Face Recognition", frame)
                pressed_key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                # The headless OpenCV build has no window support, so the preview
                # cannot open and 'q' could never stop the loop. Stop with a clear
                # message instead of crashing the request.
                return False, ("The camera window cannot open because the headless OpenCV "
                               "build is installed. Run: pip uninstall -y "
                               "opencv-contrib-python-headless opencv-python-headless "
                               "opencv-python opencv-contrib-python  then  pip install "
                               "opencv-contrib-python==4.10.0.84  and restart the app.")

            if pressed_key == ord("q"):
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if EMAIL_CONFIGURED:
        send_email()
    return True, f"Attendance captured for {len(recognized_students)} recognized doctor(s)."


# --------------------------------------------------------------------------
# Auth (signup / login)
# --------------------------------------------------------------------------
CSV_FILE = "login.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        csv.writer(file).writerow(["phc", "name", "email", "phone", "password"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    required = ["phc", "name", "email", "phone", "password"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"success": False, "message": f"Missing fields: {', '.join(missing)}"}), 400

    phc, name, email, phone, password = (data[f] for f in required)

    with open(CSV_FILE, "r", newline="") as file:
        for row in csv.reader(file):
            if row and row[2] == email:
                return "Email already exists! Please login."

    with open(CSV_FILE, "a", newline="") as file:
        csv.writer(file).writerow([phc, name, email, phone, generate_password_hash(password)])

    return "Signup successful! You can now log in."


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email, password = data.get("email"), data.get("password")
    if not email or not password:
        return "Invalid email or password."

    with open(CSV_FILE, "r", newline="") as file:
        for row in csv.reader(file):
            if row and row[2] == email:
                stored_password = row[4]
                # Supports both new hashed passwords and any pre-existing
                # plaintext rows created before this fix.
                valid = (
                    check_password_hash(stored_password, password)
                    if stored_password.startswith(("pbkdf2:", "scrypt:"))
                    else stored_password == password
                )
                if valid:
                    session["user"] = row[1]
                    return "success"

    return "Invalid email or password."


@app.route("/view")
def view():
    if "user" in session:
        return render_template("view.html", username=session["user"])
    return redirect(url_for("index"))


# ---------------------------------------------------------
# AI ASSISTANT – PHC COPILOT (RAG + live data, powered by Nexus API)
# ---------------------------------------------------------

@app.route("/ai-assistant")
def ai_assistant_page():

    # Only logged-in users can access the assistant
    if "user" not in session:
        return redirect(url_for("index"))

    return render_template("ai_assistant.html")


@app.route("/ask-ai", methods=["POST"])
def ask_ai():

    # Check user login
    if "user" not in session:
        return jsonify({
            "error": "Please log in to use the AI Assistant."
        }), 401

    # Read question sent from chatbot webpage
    data = request.get_json(silent=True) or {}

    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if len(question) > 1000:
        return jsonify({
            "error": "Question is too long. Maximum 1000 characters."
        }), 400

    try:
        # PHC Copilot: RAG over documents + live project data, one Nexus call.
        return jsonify(copilot.answer_question(question, LiveData()))

    except NexusError as error:
        # AI unavailable (no key / quota / API failure): tell the user clearly.
        # Retrieved document excerpts are returned separately and labelled as
        # such in the UI - they are NOT presented as an AI answer.
        return jsonify({
            "error": error.message,
            "excerpts": error.excerpts or []
        }), error.status

    except Exception:
        app.logger.exception("AI assistant error")

        return jsonify({
            "error": "Unable to process your question. Please try again."
        }), 500


@app.route("/ai-status")
def ai_status():
    """Small status endpoint for the assistant page / demo."""
    if "user" not in session:
        return jsonify({"error": "Please log in."}), 401
    cfg = nexus_client.settings()
    return jsonify({
        "configured": nexus_client.is_configured(),
        "model": cfg["model"],
        "usage_today": nexus_client.usage_today()
    })


@app.route("/extract-report", methods=["POST"])
def extract_report():
    """Report-to-Form: read an uploaded report and return fields that the
    browser uses to PRE-FILL the daily update form (nothing is saved here)."""
    if "user" not in session:
        return jsonify({"error": "Please log in to use this feature."}), 401

    uploaded = request.files.get("report")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Please choose a PDF or TXT report."}), 400

    uploaded.stream.seek(0, 2)
    size = uploaded.stream.tell()
    uploaded.stream.seek(0)
    if size > 5 * 1024 * 1024:
        return jsonify({"error": "File size must not exceed 5 MB."}), 400

    try:
        known = sorted({r[0] for r in db.session.query(PHCUpdate.phc_name).distinct().all() if r[0]})
        return jsonify(report_extractor.extract_fields(uploaded, known))
    except report_extractor.ReportError as error:
        return jsonify({"error": str(error)}), 422
    except NexusError as error:
        return jsonify({"error": error.message}), error.status
    except Exception:
        app.logger.exception("Report extraction failed")
        return jsonify({"error": "Could not process the report. Please try again."}), 500


# ---------------------------------------------------------
# PHC DOCUMENT UPLOAD
# ---------------------------------------------------------

@app.route("/upload-documents", methods=["GET", "POST"])
def upload_phc_document():

    # Only logged-in users can upload documents
    if "user" not in session:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("upload_documents.html")

    uploaded_file = request.files.get("document")

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "upload_documents.html",
            message="Please select a document."
        )

    # Allow only PDF and TXT files
    filename = secure_filename(uploaded_file.filename)

    if not filename or not filename.lower().endswith((".pdf", ".txt")):
        return render_template(
            "upload_documents.html",
            message="Only PDF and TXT files are allowed."
        )

    # Check file size (maximum 5 MB)
    uploaded_file.stream.seek(0, 2)
    file_size = uploaded_file.stream.tell()
    uploaded_file.stream.seek(0)

    if file_size > 5 * 1024 * 1024:
        return render_template(
            "upload_documents.html",
            message="File size must not exceed 5 MB."
        )

    # Create the PHC document folder if it does not exist
    document_folder = Path(__file__).parent / "phc_documents"
    document_folder.mkdir(exist_ok=True)

    file_path = document_folder / filename

    # Avoid overwriting an existing document
    if file_path.exists():
        return render_template(
            "upload_documents.html",
            message="A document with this filename already exists."
        )

    try:
        uploaded_file.save(file_path)

        return render_template(
            "upload_documents.html",
            message=f"{filename} uploaded successfully!"
        )

    except Exception:
        app.logger.exception("Document upload failed")

        return render_template(
            "upload_documents.html",
            message="Upload failed. Please try again."
        )


@app.route("/detect", methods=["POST"])
def detect():
    try:
        success, message = recognize_face()
    except Exception:
        app.logger.exception("Face detection failed")
        success, message = False, "Face detection failed unexpectedly. See the terminal for details."
    flash(message, "success" if success else "danger")
    return redirect(url_for("index"))


# --------------------------------------------------------------------------
# PHC-wise staff + attendance
# staff.csv (ID, Name, Role, PHC) says which PHC each face-recognition person
# belongs to. It is created automatically from names.json; the PHC/role are set
# on the Attendance page. attendance.csv itself is unchanged - the PHC is looked
# up from staff.csv when the records are shown, so old records are covered too.
# --------------------------------------------------------------------------
staff_file = "staff.csv"
STAFF_ROLES = ["Doctor", "Staff"]


def _id_key(value):
    """Normalise an ID read from a CSV ('1', 1, 1.0) to the string '1'."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def save_staff(staff):
    with open(staff_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Role", "PHC"])
        for sid in sorted(staff, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k)):
            s = staff[sid]
            writer.writerow([sid, s["name"], s["role"], s["phc"]])


def load_staff():
    """{id: {id, name, role, phc}}. Everyone in names.json is added automatically."""
    staff = {}
    if os.path.exists(staff_file):
        df = pd.read_csv(staff_file, dtype=str).fillna("")
        for _, r in df.iterrows():
            sid = _id_key(r.get("ID", ""))
            if sid:
                staff[sid] = {"id": sid, "name": r.get("Name", "").strip(),
                              "role": r.get("Role", "").strip() or "Doctor",
                              "phc": r.get("PHC", "").strip()}
    changed = not os.path.exists(staff_file)
    try:
        with open(names_file, "r") as f:
            names = json.load(f)
    except (OSError, json.JSONDecodeError):
        names = {}
    for sid, name in names.items():
        sid = _id_key(sid)
        if sid not in staff:
            staff[sid] = {"id": sid, "name": str(name).strip(), "role": "Doctor", "phc": ""}
            changed = True
    if changed:
        save_staff(staff)
    return staff


def known_phc_names():
    """Every PHC name used anywhere (daily updates, patients, staff)."""
    names = {r[0] for r in db.session.query(PHCUpdate.phc_name).distinct().all() if r[0]}
    names |= {r[0] for r in db.session.query(Patient.phc_name).distinct().all() if r[0]}
    names |= {s["phc"] for s in load_staff().values() if s["phc"]}
    return sorted(names)


@app.route("/attendance")
def view_attendance():
    staff = load_staff()
    selected = (request.args.get("phc") or "").strip()
    today = datetime.today().strftime("%Y-%m-%d")

    df = pd.read_csv(attendance_file)
    records, present_today = [], set()
    if not df.empty:
        # Newest attendance first (records are appended in time order in the CSV).
        df["_when"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("_when", ascending=False, na_position="last", kind="stable")
        for _, r in df.iterrows():
            sid = _id_key(r["ID"])
            info = staff.get(sid, {})
            stamp = str(r["Timestamp"])
            if stamp.startswith(today):
                present_today.add(sid)
            records.append({
                "id": sid, "name": str(r["Name"]).strip(),
                "role": info.get("role", "-"),
                "phc": info.get("phc") or UNASSIGNED_PHC,
                "engagement": r["Engagement"], "location": r["Location"],
                "timestamp": stamp,
            })

    if selected:
        records = [x for x in records if x["phc"] == selected]
    scope = [s for s in staff.values()
             if not selected or (s["phc"] or UNASSIGNED_PHC) == selected]
    summary = {
        "date": today,
        "total": len(scope),
        "present": sum(1 for s in scope if s["id"] in present_today),
        "absent": [s["name"] for s in scope if s["id"] not in present_today],
    }
    phcs = sorted(set(known_phc_names()))
    return render_template(
        "attendance.html", records=records, selected=selected, phcs=phcs,
        summary=summary, staff_rows=sorted(staff.values(), key=lambda s: (s["phc"] == "", s["phc"].lower(), s["name"].lower())),
        roles=STAFF_ROLES, unassigned=UNASSIGNED_PHC, logged_in="user" in session)


@app.route("/attendance/assign", methods=["POST"])
def assign_staff_phc():
    """Set the PHC and role of one doctor/staff member (login required)."""
    if "user" not in session:
        return redirect(url_for("index"))
    sid = _id_key(request.form.get("id", ""))
    staff = load_staff()
    back = request.form.get("filter") or None
    if sid not in staff:
        flash("Unknown staff ID.", "danger")
        return redirect(url_for("view_attendance", phc=back))
    role = request.form.get("role")
    if role not in STAFF_ROLES:
        role = staff[sid]["role"]
    phc = " ".join((request.form.get("phc") or "").split())[:100]
    if phc.lower() == UNASSIGNED_PHC.lower():
        phc = ""
    for known in known_phc_names():  # keep one consistent spelling per PHC
        if known.lower() == phc.lower():
            phc = known
            break
    staff[sid].update(role=role, phc=phc)
    save_staff(staff)
    flash(f"{staff[sid]['name']} saved: {role}, {phc or UNASSIGNED_PHC}.", "success")
    return redirect(url_for("view_attendance", phc=back))


# --------------------------------------------------------------------------
# PHC daily updates + patient records + anomaly detection
# --------------------------------------------------------------------------
class PHCUpdate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phc_name = db.Column(db.String(100), nullable=False)
    patient_count = db.Column(db.Integer, nullable=False)
    diagnosis = db.Column(db.Text, nullable=False)
    treatment = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    phone = db.Column(db.String(20))
    diagnosis = db.Column(db.Text)
    treatment = db.Column(db.Text)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    # NEW: which PHC submitted this patient. Without this column there was
    # no link at all between a patient record and a PHC, so the dashboard
    # could only ever show one flat list of everybody.
    phc_name = db.Column(db.String(100))


# Label used on the dashboard for patient rows saved before phc_name existed.
UNASSIGNED_PHC = "Not assigned to a PHC"


class LiveData:
    """Read-only, privacy-safe views of the live project data used by the
    PHC Copilot. Only aggregates are returned - never patient names/phones."""

    def phc_names(self):
        names = {r[0] for r in db.session.query(PHCUpdate.phc_name).distinct().all() if r[0]}
        names |= {s["phc"] for s in load_staff().values() if s["phc"]}
        return sorted(names)

    def phc_trends(self, phcs=None):
        """Per-PHC trend from the last (up to) 7 daily updates, newest first.
        Also reports the dates, so answers never call old data 'this week'."""
        out = []
        today = datetime.now().date()
        for phc in (phcs or self.phc_names()):
            rows = (PHCUpdate.query.filter_by(phc_name=phc)
                    .order_by(PHCUpdate.date.desc()).limit(7).all())
            counts = [r.patient_count for r in rows]  # newest first
            if not rows:
                continue
            latest_date = rows[0].date.date() if hasattr(rows[0].date, "date") else today
            days_old = (today - latest_date).days
            base = {"phc": phc, "n": len(counts), "counts": counts, "latest_date": latest_date.isoformat(),
                    "days_old": days_old, "oldest_date": (rows[-1].date.date() if hasattr(rows[-1].date, "date") else latest_date).isoformat(),
                    "diagnosis": (rows[0].diagnosis or "").replace("\r", " ").replace("\n", ", ")[:80]}
            if len(counts) < 2:
                out.append(dict(base, status="insufficient", latest=counts[0], avg=0))
                continue
            latest, avg = counts[0], mean(counts[1:])
            status = "normal"
            if latest < avg * 0.5:
                status = "drop"
            elif latest > avg * 1.5:
                status = "rise"
            elif len(counts) >= 4 and stdev(counts[1:]) > avg * 0.3:
                # needs at least 3 earlier points; with fewer, variation is just noise
                status = "fluctuating"
            out.append(dict(base, status=status, latest=latest, avg=avg))
        return out

    def attendance_today(self, phc=None):
        """Who marked attendance today, from attendance.csv (what the Attendance
        page shows). With `phc`, only that PHC's doctors/staff (from staff.csv)
        are counted. 'Registered' staff are the people the face-recognition model
        knows, because those are the people who can mark attendance."""
        if not os.path.exists(attendance_file):
            return None
        today = datetime.today().strftime("%Y-%m-%d")
        att = pd.read_csv(attendance_file)
        rows = att[att["Timestamp"].astype(str).str.startswith(today)].copy()
        rows = rows.sort_values("Timestamp", ascending=False)
        staff = load_staff()

        def in_scope(sid):
            return not phc or (staff.get(sid, {}).get("phc") or UNASSIGNED_PHC) == phc

        present, seen = [], set()
        for _, r in rows.iterrows():
            key = _id_key(r["ID"])
            if key in seen:
                continue
            seen.add(key)
            if in_scope(key):
                present.append({"id": key, "name": str(r["Name"]).strip(),
                                "time": str(r["Timestamp"])[11:16]})
        registered = {sid: s["name"] for sid, s in staff.items() if in_scope(sid)}
        absent = [name for sid, name in registered.items() if sid not in seen]
        return {"date": today, "present": present, "absent": absent,
                "total": len(registered) or len(present), "phc": phc}

    def diagnosis_summary(self, phcs=None):
        from collections import Counter
        q = PHCUpdate.query
        if phcs:
            q = q.filter(PHCUpdate.phc_name.in_(phcs))
        counter = Counter()
        for row in q.all():
            for term in re.split(r"[,\n\r;]+", row.diagnosis or ""):
                term = term.strip().lower()
                if term:
                    counter[term] += 1
        return counter.most_common(6)

    def patient_stats(self, phcs=None):
        from collections import Counter
        q = Patient.query
        if phcs:
            q = q.filter(Patient.phc_name.in_(phcs))
        rows = q.all()
        by_phc = Counter((r.phc_name or UNASSIGNED_PHC) for r in rows)
        return {"total": len(rows), "by_phc": by_phc.most_common(6)}


def send_alert(email, message):
    if not EMAIL_CONFIGURED:
        print(f"⚠ Email not configured, skipping alert: {message}")
        return
    msg = MIMEText(message)
    msg["Subject"] = "🚨 Healthcare Anomaly Alert"
    msg["From"] = SMTP_USERNAME
    msg["To"] = email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, email, msg.as_string())
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


@app.route("/submit_update", methods=["GET", "POST"])
def submit_update():
    """Handles BOTH forms on submit_update.html: 'Submit Patient Data' and
    'Submit PHC Daily Update'. The original code only understood the daily
    update's fields and crashed with a 500 BadRequestKeyError whenever the
    patient-data form was submitted, since it unconditionally read
    request.form['phc_name']. This now branches on which fields are present.
    """
    if request.method == "POST":
        if "patient_name" in request.form:
            new_patient = Patient(
                name=request.form["patient_name"],
                age=int(request.form.get("age") or 0),
                phone=request.form.get("phno", ""),
                diagnosis=request.form.get("diagnosis", ""),
                treatment=request.form.get("treatment", ""),
                # NEW: the patient form now carries the PHC this record
                # belongs to, so the dashboard can group by it.
                phc_name=(request.form.get("phc_name") or "").strip() or None,
            )
            db.session.add(new_patient)
            db.session.commit()

            # Keep patients.csv (already part of the project) in sync.
            # CHANGED: rewritten from the database each time instead of
            # appended to. The old append path could not add the new PHC
            # column to a file whose header was already written without it,
            # which would have left ragged rows.
            save_patients_to_csv()

            flash("✅ Patient data submitted successfully.", "success")
            return redirect(url_for("submit_update"))

        elif "phc_name" in request.form:
            new_entry = PHCUpdate(
                phc_name=request.form["phc_name"],
                patient_count=int(request.form.get("patient_count") or 0),
                diagnosis=request.form.get("diagnosis", ""),
                treatment=request.form.get("treatment", ""),
            )
            db.session.add(new_entry)
            db.session.commit()
            return redirect(url_for("check_alerts"))

        flash("⚠️ Unrecognized form submission.", "warning")
        return redirect(url_for("submit_update"))

    # NEW: feed the PHC names already in the database to the patient form's
    # dropdown, so records get grouped under a consistent spelling instead
    # of "Govt PHC Salem" / "Government PHC salem" splitting into two cards.
    known_phcs = sorted(
        {row[0] for row in db.session.query(PHCUpdate.phc_name).distinct().all() if row[0]}
        | {row[0] for row in db.session.query(Patient.phc_name).distinct().all() if row[0]}
    )
    return render_template("submit_update.html", known_phcs=known_phcs)


@app.route("/check_alerts")
def check_alerts():
    phc_names = [row[0] for row in db.session.query(PHCUpdate.phc_name).distinct().all()]
    anomalies_detected = False

    for phc in phc_names:
        updates = PHCUpdate.query.filter_by(phc_name=phc).order_by(PHCUpdate.date.desc()).limit(7).all()
        patient_counts = [u.patient_count for u in updates]

        if len(patient_counts) < 2:
            flash(f"⚠️ Not enough data for {phc} to analyze trends.", "warning")
            continue

        # patient_counts is newest-first: [0] is the latest update, the rest is the baseline
        # (the old code compared the OLDEST count to the average, giving false alerts).
        latest_count = patient_counts[0]
        avg_count = mean(patient_counts[1:])
        std_dev = stdev(patient_counts[1:]) if len(patient_counts) > 2 else 0

        if latest_count < (avg_count * 0.5):
            message = f"🚨 Anomaly detected at {phc}! Sudden drop in patient count: {latest_count} (Expected: {avg_count:.2f})"
            send_alert(ADMIN_EMAIL, message)
            flash(message, "danger")
            anomalies_detected = True
        elif latest_count > (avg_count * 1.5):
            message = f"⚠️ Possible outbreak at {phc}! Sudden rise in patient count: {latest_count} (Expected: {avg_count:.2f})"
            send_alert(ADMIN_EMAIL, message)
            flash(message, "warning")
            anomalies_detected = True
        elif std_dev > (avg_count * 0.3):
            message = f"⚠️ Unusual fluctuations detected at {phc}! High variation in patient counts over the past week."
            send_alert(ADMIN_EMAIL, message)
            flash(message, "warning")
            anomalies_detected = True

    if not anomalies_detected:
        flash("✅ No anomalies detected in any PHC.", "success")

    return redirect(url_for("dashboard"))


def save_patients_to_csv():
    """Exports every Patient row to patients.csv, PHC column included."""
    with open("patients.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Age", "Phone", "Diagnosis", "Treatment", "PHC", "Date"])
        for p in Patient.query.order_by(Patient.date.asc()).all():
            writer.writerow([
                p.name, p.age, p.phone, p.diagnosis, p.treatment,
                p.phc_name or "",
                p.date.strftime("%Y-%m-%d") if p.date else "",
            ])
    return "patients.csv"


def save_to_csv():
    """Exports PHCUpdate data to phcdetails.csv."""
    updates = PHCUpdate.query.order_by(PHCUpdate.date.desc()).all()
    file_path = "phcdetails.csv"
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "PHC Name", "Patient Count", "Diagnosis", "Treatment", "Date"])
        for update in updates:
            writer.writerow([update.id, update.phc_name, update.patient_count,
                              update.diagnosis, update.treatment, update.date])
    return file_path


@app.route("/dashboard")
def dashboard():
    updates = PHCUpdate.query.order_by(PHCUpdate.date.desc()).all()
    patients = Patient.query.order_by(Patient.date.desc()).all()

    # ---------------------------------------------------------
    # PHC GROUPING
    # CHANGED: patients used to be rendered as one flat "Submitted
    # Patients" table underneath the PHC list. They are now bucketed by
    # phc_name and handed to the template as one entry per PHC, so each
    # PHC card shows only its own daily updates and its own patients.
    # ---------------------------------------------------------
    groups = {}

    def bucket(name):
        return groups.setdefault(name, {
            "name": name,
            "updates": [],
            "patients": [],
            "total_patient_count": 0,
        })

    for update in updates:
        g = bucket(update.phc_name)
        g["updates"].append(update)
        g["total_patient_count"] += update.patient_count or 0

    for patient in patients:
        # Rows saved before the phc_name column existed have no PHC, so
        # they land in a clearly-labelled bucket instead of vanishing.
        bucket(patient.phc_name or UNASSIGNED_PHC)["patients"].append(patient)

    # Real PHCs first (alphabetical), the unassigned bucket last.
    phc_groups = sorted(
        groups.values(),
        key=lambda g: (g["name"] == UNASSIGNED_PHC, g["name"].lower()),
    )

    # ---------------------------------------------------------
    # BAR CHART DATA
    # Patient count comparison between different PHCs
    # ---------------------------------------------------------
    phc_counts_dict = {}

    for update in updates:
        phc_counts_dict[update.phc_name] = (
            phc_counts_dict.get(update.phc_name, 0)
            + update.patient_count
        )

    # ---------------------------------------------------------
    # LINE CHART DATA
    # Total patient count for each day
    # ---------------------------------------------------------
    daily_counts = {}

    for update in updates:
        if update.date:
            date_key = update.date.strftime("%Y-%m-%d")

            daily_counts[date_key] = (
                daily_counts.get(date_key, 0)
                + update.patient_count
            )

    # Sort dates from oldest to newest
    trend_labels = sorted(daily_counts.keys())

    trend_counts = [
        daily_counts[date]
        for date in trend_labels
    ]

    return render_template(
        "dashboard.html",
        updates=updates,
        patients=patients,
        phc_groups=phc_groups,
        unassigned_phc=UNASSIGNED_PHC,

        # Bar chart
        phc_labels=list(phc_counts_dict.keys()),
        phc_counts=list(phc_counts_dict.values()),

        # Line chart
        trend_labels=trend_labels,
        trend_counts=trend_counts
    )

phc_file = "phcdetails.csv"


@app.route("/phc-email", methods=["POST"])
def phc_email():
    save_to_csv()  # keep the export current before emailing it
    return _send_csv_email(
        "📊 PHC Report",
        "Please find the attached PHC report for today.",
        phc_file,
        "PHC details file not found!",
    )


# --------------------------------------------------------------------------
# Doctor absenteeism alerts (email + SMS), run on a schedule
# --------------------------------------------------------------------------
doctors_file = "doctors.csv"


def check_doctor_attendance():
    """Check attendance and send alerts to absent doctors."""
    today = datetime.today().strftime("%Y-%m-%d")  # FIXED: was datetime.datetime.today()

    if not os.path.exists(doctors_file):
        print(f"⚠ {doctors_file} not found, skipping attendance check.")
        return

    doctors_df = pd.read_csv(doctors_file)
    attendance_df = pd.read_csv(attendance_file)

    for _, doctor in doctors_df.iterrows():
        doctor_id, doctor_name = doctor["ID"], str(doctor["Name"]).strip()
        email, phone = doctor["Email"], doctor["Phone"]

        print(f"👨‍⚕️ Checking {doctor_name} (ID: {doctor_id}) - Email: {email}, Phone: {phone}")

        doctor_attendance = attendance_df[
            (attendance_df["ID"].astype(str) == str(doctor_id))
            & (attendance_df["Timestamp"].astype(str).str.startswith(today))
        ]

        if doctor_attendance.empty:
            print(f"🚨 ALERT: {doctor_name} (ID: {doctor_id}) did NOT mark attendance. Sending email & SMS.")
            message = (
                f"Dear {doctor_name}, you have not marked attendance today.\n\n"
                "Please confirm your status:\n"
                "- I will come: https://yourapp.com/confirm-attendance\n"
                "- I am on leave: https://yourapp.com/apply-leave"
            )
            send_alert_email(email, "Attendance Reminder", message)
            send_sms(phone, message)
        else:
            print(f"✅ {doctor_name} (ID: {doctor_id}) has already marked attendance.")


def send_alert_email(to_email, subject, message):
    if not EMAIL_CONFIGURED:
        print(f"⚠ Email not configured, skipping alert email to {to_email}.")
        return
    try:
        msg = MIMEText(message, "plain")
        msg["Subject"] = subject
        msg["From"] = SMTP_USERNAME
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, to_email, msg.as_string())
        print(f"📧 Email sent successfully to {to_email}")
    except Exception as e:
        print(f"❌ Email error: {e}")


def send_sms(to_phone, message):
    if not SMS_CONFIGURED:
        print(f"⚠ SMS not configured, skipping SMS to {to_phone}.")
        return
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone,  # FIXED: was a hardcoded phone number, ignoring the doctor's real number
        )
        print(f"📱 SMS sent successfully to {to_phone}")
    except Exception as e:
        print(f"❌ SMS error: {e}")
    # FIX: send_sms was defined twice (a leftover duplicate silently shadowed
    # this first one - harmless since they were identical, but removed to
    # avoid confusion).


def run_scheduler():
    """Check the time every minute and run the attendance check at 9 PM."""
    last_run_date = None
    while True:
        now = datetime.now()
        # FIXED: original code compared a full "%Y-%m-%d %H:%M:%S" timestamp
        # to the literal string "12:35", which could never match, so the
        # check never ran. Now compares HH:MM only, once per day.
        if now.strftime("%H:%M") == "21:00" and last_run_date != now.date():
            print("📅 Running attendance check at 9 PM...")
            check_doctor_attendance()
            last_run_date = now.date()
        time.sleep(60)


scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Create tables on import (not just when run as __main__), so `flask run`
# and WSGI servers (gunicorn, etc.) also get a working database.
with app.app_context():
    db.create_all()

    # NEW: lightweight migration. db.create_all() creates missing TABLES but
    # never adds a missing COLUMN to a table that already exists, so an
    # existing healthcare.db would keep its old `patient` table and every
    # query would fail with "no such column: patient.phc_name". This adds
    # the column once, in place, leaving existing rows untouched (their
    # phc_name is NULL and they show under UNASSIGNED_PHC on the dashboard).
    try:
        existing_columns = {
            row[1] for row in db.session.execute(db.text("PRAGMA table_info(patient)"))
        }
        if existing_columns and "phc_name" not in existing_columns:
            db.session.execute(db.text("ALTER TABLE patient ADD COLUMN phc_name VARCHAR(100)"))
            db.session.commit()
            print("✅ Added patient.phc_name column to the existing database.")
    except Exception as e:  # non-SQLite backends / permissions
        db.session.rollback()
        print(f"⚠ Could not verify the patient.phc_name column automatically: {e}")

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=debug_mode, host=host, port=port, use_reloader=debug_mode)
