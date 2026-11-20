"""
SmartPHC Copilot - RAG over PHC documents + LIVE project data.

Answers are short, direct and natural. Numbers, averages and counts are only
given when the user asks for them (statistics, counts, comparisons, reports).
Patient names and phone numbers are never sent to the API.
"""
import re
from datetime import date

import ai_assistant
import nexus_client

MAX_LIVE_CHARS = 1600
MAX_DOC_CHARS = 1800
DOC_MIN_SCORE = 0.05
DOC_MIN_SCORE_WITH_LIVE = 0.15

SYSTEM_PROMPT = (
    "You are SmartPHC Copilot, a friendly assistant for Primary Health Centre (PHC) "
    "administrators. Reply in English unless the question is in another language. "
    "Use ONLY the LIVE DATA and DOCUMENT EXCERPTS provided; never invent facts. "
    "Answer the question directly in one to three natural sentences, like a helpful "
    "colleague. Do NOT use headings, bullet points, labels such as 'Key facts', or "
    "mention sources or data files. Do NOT quote numbers, averages, counts or dates "
    "unless the user explicitly asks for statistics, counts, comparisons, details or "
    "a report; if they do, give exactly the relevant figures. "
    "Say there is insufficient recent data ONLY if the user asks about 'this week', "
    "'today' or 'now' AND the latest update is more than 7 days old; otherwise answer "
    "from the data. "
    "If the answer is not in the data, say you do "
    "not have that information. Each PHC count is one daily update, never a 'week'. "
    "You are not a diagnostic tool; give no individual medical advice. Maximum 60 words."
)

STATUS_WORD = {"normal": "NORMAL", "drop": "DROP", "rise": "RISE",
               "fluctuating": "FLUCTUATING", "insufficient": "NOT ENOUGH DATA"}

INTENT_KEYWORDS = {
    "attendance": ("attendance", "absent", "present", "doctor", "staff", "late",
                   "leave", "face"),
    "trends": ("count", "patients", "trend", "anomaly", "anomalies", "unusual",
               "outbreak", "drop", "rise", "increase", "decrease", "alert",
               "latest", "this week", "busiest", "daily", "update"),
    "diagnosis": ("diagnosis", "diagnoses", "disease", "diseases", "illness",
                  "common", "cases", "fever", "cold", "cough", "infection"),
    "patients": ("patient records", "registered", "age", "how many patients",
                 "total patients"),
    "overview": ("summary", "summarize", "summarise", "overview", "status",
                 "report", "all phc"),
}

DETAIL_RE = re.compile(
    r"\b(statistics?|stats?|counts?|numbers?|figures?|average|how many|compar\w*|"
    r"details?|detailed|report|breakdown|total|exact|data)\b", re.I)

LOW_RE = re.compile(r"\b(low\w*|least|fewest|less|minimum|smallest)\b", re.I)
HIGH_RE = re.compile(r"\b(high\w*|most|busiest|maximum|more|largest|biggest)\b", re.I)


def wants_detail(question):
    return bool(DETAIL_RE.search(question))


def detect_intents(question, phc_names):
    q = question.lower()
    intents = set()
    for name, words in INTENT_KEYWORDS.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"(?:e?s|d|ed)?\b", q):
                intents.add(name)
                break
    mentioned = [p for p in phc_names if p.lower() in q]
    if not mentioned:  # allow "Karur" for "Government PHC Karur"
        for p in phc_names:
            tail = p.lower().split()[-1]
            if len(tail) > 3 and re.search(r"\b" + re.escape(tail) + r"\b", q):
                mentioned.append(p)
    if mentioned and not intents:
        intents.add("trends")
    if "overview" in intents:
        intents.update({"trends", "attendance", "diagnosis"})
    return intents, mentioned


def _trim(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def attendance_sentence(a):
    n_total = a["total"]
    where = f" at {a['phc']}" if a.get("phc") else ""
    if a.get("phc") and not a["present"] and not a["absent"]:
        return f"No doctors or staff are assigned to {a['phc']} yet (set this on the Attendance page)."
    if not a["present"]:
        return (f"No attendance has been marked today{where} yet; "
                f"0 of {n_total} registered staff are present.")
    who = ", ".join(f"{p['name']} ({p['time']})" for p in a["present"])
    text = (f"Today{where}, {len(a['present'])} of {n_total} registered staff "
            f"are marked present: {who}.")
    if a["absent"]:
        text += " Not marked yet: " + ", ".join(a["absent"]) + "."
    return text


def build_live_context(intents, phcs, live):
    """Compact plain-text facts from the real database / CSV files (for the model)."""
    lines = []
    if "trends" in intents:
        for t in live.phc_trends(phcs):
            when = f"{t['latest_date']} ({t['days_old']} days ago)"
            if t["status"] == "insufficient":
                lines.append(f"- {t['phc']}: only {t['n']} update(s), latest on {when}; "
                             f"status NOT ENOUGH DATA.")
            else:
                line = (f"- {t['phc']}: latest update {when} count {t['latest']} vs average "
                        f"{t['avg']:.1f} of the previous {t['n'] - 1} updates -> status "
                        f"{STATUS_WORD[t['status']]}")
                if t["diagnosis"]:
                    line += f"; latest diagnosis: {t['diagnosis']}"
                lines.append(line)
    if "attendance" in intents:
        for p in (phcs or [None]):
            a = live.attendance_today(p)
            if a:
                lines.append("- " + attendance_sentence(a))
    if "diagnosis" in intents:
        d = live.diagnosis_summary(phcs)
        if d:
            lines.append("- Most frequent diagnoses in daily updates: "
                         + ", ".join(f"{k} ({v})" for k, v in d) + ".")
    if "patients" in intents:
        s = live.patient_stats(phcs)
        if s:
            lines.append(f"- Patient records: {s['total']} total; by PHC: "
                         + ", ".join(f"{k} {v}" for k, v in s["by_phc"]) + ".")
    return _trim("\n".join(lines), MAX_LIVE_CHARS)


def _result(answer, used_ai=False, live_data=None, sources=None, cached=False, tokens=0):
    # "facts" is always empty: the page no longer shows a facts box.
    return {"answer": answer, "facts": [], "sources": sources or [],
            "live_data": live_data or [], "cached": cached,
            "tokens": tokens, "used_ai": used_ai}


def _single_phc_trend(phc, question, live):
    """One PHC, trend question: a direct sentence written by the code (no AI call).
    Numbers are added only if the user asks for them."""
    q = question.lower()
    trend = next(iter(live.phc_trends([phc])), None)
    if not trend:
        return _result(f"No daily updates have been recorded for {phc} yet.",
                       live_data=["trends"])

    detail = wants_detail(question)
    period = "this week" if "this week" in q else "recently"
    status = trend["status"]
    stale = trend["days_old"] > 7
    asks_now = bool(re.search(r"\b(this week|today|now)\b", q))
    when = " this week" if "this week" in q else ""

    detail_text = ""
    if status != "insufficient":
        detail_text = (f" The latest count is {trend['latest']}, against an average of "
                       f"{trend['avg']:.1f} over the previous {trend['n'] - 1} updates.")
        if trend["diagnosis"]:
            detail_text += f" The latest diagnosis recorded is {trend['diagnosis']}."

    if status == "insufficient":
        answer = (f"There is insufficient data to determine whether {phc} has had "
                  f"unusual activity {period}.")
        if detail:
            answer += f" Only {trend['n']} update has been recorded."
    elif stale and asks_now:
        if detail:
            answer = (f"The latest update for {phc} is from {trend['latest_date']} "
                      f"({trend['days_old']} days ago), so it is not recent data."
                      + detail_text)
        else:
            answer = (f"There is insufficient recent data to determine whether {phc} "
                      f"has experienced unusual activity {period}.")
    else:
        if status == "normal":
            answer = f"No unusual change at {phc}{when}. Patient numbers look normal."
        elif status == "drop":
            answer = f"Yes, there is a drop in patient numbers at {phc}{when}."
        elif status == "rise":
            answer = f"Yes, there is a rise in patient numbers at {phc}{when}."
        else:
            answer = (f"Patient numbers at {phc} are fluctuating{when}, "
                      "so activity is somewhat irregular.")
        if stale:
            answer += f" This is based on the latest update, from {trend['latest_date']}."
        if detail:
            answer += detail_text
    return _result(answer, live_data=["trends"])


def is_ranking_question(question):
    q = question.lower()
    return bool(re.search(r"\b(which|what)\b", q) and re.search(r"\bphcs?\b", q)
                and (LOW_RE.search(q) or HIGH_RE.search(q)))


def _phc_level(t):
    """Average of the recorded daily counts (falls back to the latest count)."""
    counts = [c for c in (t.get("counts") or []) if c is not None]
    if counts:
        return sum(counts) / len(counts), len(counts)
    if t.get("latest") is not None:
        return float(t["latest"]), 1
    return None, 0


def _rank_phcs(question, live):
    """Which PHC is busiest / has the lowest patient numbers: answered from the data."""
    rows = []
    for t in live.phc_trends(None):
        level, n = _phc_level(t)
        if level is not None:
            rows.append((t["phc"], level, n))
    if not rows:
        return _result("No patient counts have been recorded for any PHC yet.",
                       live_data=["trends"])
    low = bool(LOW_RE.search(question)) or not HIGH_RE.search(question)
    rows.sort(key=lambda r: r[1], reverse=not low)
    phc, level, n = rows[0]
    value = f"{level:.0f}" if abs(level - round(level)) < 0.05 else f"{level:.1f}"
    word = "lowest" if low else "highest"
    if n > 1:
        answer = (f"{phc} has the {word} patient numbers, averaging {value} "
                  f"patients per daily update.")
    else:
        answer = f"{phc} has the {word} patient count, with {value} patients."
    return _result(answer, live_data=["trends"])


def answer_question(question, live):
    phc_names = live.phc_names()
    intents, mentioned = detect_intents(question, phc_names)

    if not mentioned and is_ranking_question(question):
        return _rank_phcs(question, live)

    if intents == {"trends"} and len(mentioned) == 1:
        return _single_phc_trend(mentioned[0], question, live)

    if intents == {"attendance"}:
        atts = [live.attendance_today(p) for p in (mentioned or [None])]
        text = ("\n".join(attendance_sentence(x) for x in atts if x) or
                "No attendance records have been saved yet.")
        return _result(text, live_data=["attendance"])

    live_text = build_live_context(intents, mentioned or None, live) if intents else ""

    docs = ai_assistant.search_documents(question, top_k=3)
    threshold = DOC_MIN_SCORE_WITH_LIVE if live_text else DOC_MIN_SCORE
    relevant = [d for d in docs if d["score"] >= threshold]
    doc_text = _trim("".join(f"[Doc: {d['source']}] {d['text']}\n" for d in relevant),
                     MAX_DOC_CHARS)

    if not live_text and not doc_text:
        return _result("I couldn't find anything relevant to that in the PHC data or "
                       "uploaded documents. Try mentioning a PHC name, attendance, "
                       "patient counts or diagnoses, or upload a related document.")

    user_msg = f"Today: {date.today().isoformat()}\n"
    if live_text:
        user_msg += f"LIVE DATA:\n{live_text}\n"
    if doc_text:
        user_msg += f"DOCUMENT EXCERPTS:\n{doc_text}\n"
    user_msg += f"QUESTION: {question}\n"
    if wants_detail(question):
        user_msg += "(The user asked for figures: include the relevant numbers.)"
    else:
        user_msg += "(No figures requested: answer without numbers, averages or dates.)"

    try:
        result = nexus_client.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ], temperature=0)
    except nexus_client.NexusError as err:
        err.excerpts = [{"source": d["source"], "text": d["text"]} for d in relevant]
        raise

    answer = re.sub(r"(previous\s+\d+)\s+weeks?", r"\1 updates", result["text"], flags=re.I)
    usage = result.get("usage") or {}
    return _result(answer, used_ai=True,
                   live_data=sorted(intents) if live_text else [],
                   sources=sorted({d["source"] for d in relevant}),
                   cached=result["cached"],
                   tokens=int(usage.get("total_tokens") or 0))
