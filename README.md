# Centralized-PHC-Monitoring-and-Management-System

The Centralized PHC Monitoring and Management System is an advanced healthcare solution designed to optimize the operations of Primary Health Centers (PHCs), Upgraded PHCs, and Subcenters by integrating them with the District Deputy Director of Health Services (DDHS) for real-time monitoring. The system leverages machine learning, automation, and data analytics to enhance healthcare efficiency, accountability, and transparency. One of its key features is facial recognition-based doctor attendance tracking, ensuring accurate attendance records while generating automated absenteeism alerts to prevent workforce shortages. Additionally, the system enables real-time patient record management, maintaining comprehensive patient details, treatments, and diagnoses. These records are automatically analyzed and presented in graphical formats, allowing healthcare administrators to track trends and make informed decisions.
To enhance operational efficiency, the system automates daily and monthly report generation, sending structured reports in CSV format to the DDHS. A crucial component is the anomaly detection mechanism, which identifies sudden fluctuations in patient numbers at PHCs and immediately notifies healthcare officials for prompt action. This feature helps in resource allocation, emergency handling, and service optimization. By integrating advanced facial recognition, automated alerts, centralized data management, and anomaly detection, this system revolutionizes healthcare monitoring, improves service delivery, and strengthens decision-making processes in PHCs. It reduces manual workload, ensures real-time updates, and fosters a data-driven healthcare approach, ultimately enhancing patient care and healthcare administration.


---

# SmartPHC AI (Nexus API) - PHC Copilot + Report-to-Form

Two AI features were added without changing the existing modules:

1. **PHC Copilot** (`/ai-assistant`) - RAG over your uploaded PHC documents
   (PDF/TXT in `phc_documents/`) **plus live project data** (patient-count
   trends and anomaly status per PHC, today's doctor attendance, common
   diagnoses, patient totals). Ask e.g. "Is Government PHC Karur unusual this
   week?", "Which doctors are absent today?", "At what age is the measles
   vaccine given?". Each answer shows its sources and token usage.
2. **Report-to-Form** (`/submit_update` -> Daily Update form) - upload a PHC
   daily report (text PDF or TXT); the AI pre-fills the form. You review the
   fields and press Submit - nothing is saved automatically.

## How it works (architecture)

```
Browser (ai_assistant.html / submit_update.html)
   |  fetch /ask-ai  or  /extract-report        (no API key in the browser)
   v
Flask app.py  -- login check, validation
   |-- copilot.py            local router -> picks live data + doc chunks
   |     |-- LiveData (app.py)    read-only aggregates from SQLite/CSV
   |     '-- ai_assistant.py      TF-IDF retrieval, index cached on disk changes
   |-- report_extractor.py   masks phone/e-mail, extracts text locally
   '-- nexus_client.py       ONE short call, cache, token cap, daily limit
            |
            v
      Nexus API  (OpenAI-compatible /chat/completions, key from .env)
```

Token saving: retrieval, routing and statistics run locally (free); one call
per question with `max_tokens` 350 (200 for extraction); identical questions
are answered from an on-disk cache (0 tokens); questions with nothing relevant
make no API call; a local daily limit protects the free quota. Patient names
and phone numbers are never sent.

If the key is missing, the quota is exhausted or the API fails, the app shows a
clear error (document excerpts, if any, are labelled as *not* an AI answer).
Existing features keep working.

## Setup on Windows (VS Code terminal)

Use Python 3.10-3.12.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Open `.env` and set your key (get it from your Nexus dashboard):

```
NEXUS_API_KEY=paste_your_real_key_here
```

`NEXUS_BASE_URL` (default `https://nexusapi.navigatelabs.ai/v1`) and `NEXUS_MODEL`
(default `gemini-2.5-flash`) are for the Navigate Labs Nexus Unified API. Run `python check_nexus.py` to verify them. **If your Nexus
provider or model differs, change these two lines.** Then:

```
python app.py
```

Open http://127.0.0.1:5000 , sign up / log in, then click **AI Assistant** on
the main page.

## Quick demo script

1. Open **AI Assistant**, ask: *Which doctors are absent today?* (live data).
2. Ask: *What services does a PHC provide?* (documents). Upload your own
   guideline PDF/TXT via "Upload documents" first for better answers.
3. Ask the same question again -> "Cached answer (0 tokens)".
4. PHC Monitoring -> Submit Daily Update -> **Extract with AI** with a report
   file -> review the pre-filled fields -> Submit Update.

## Tests

`python tests/run_checks.py` runs 28 automated checks against a local
**stand-in** server that imitates the Nexus format (no key needed). It verifies
requests, caching, limits, error handling and privacy - it does not test your
real Nexus account; do the demo script above with your real key for that.

## Known limitations

- Needs internet and a valid Nexus key for AI answers. Free-tier limits depend
  on your Nexus account; check them on the Nexus dashboard.
- Report-to-Form reads text PDFs/TXT only (no scanned images / OCR).
- Answer quality depends on the documents you upload to `phc_documents/`.
- Face recognition needs a webcam (unchanged).
