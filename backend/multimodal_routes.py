import os
import sys
import datetime
import random
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from flask import send_file
import tempfile
import datetime
from flask import Blueprint, request, jsonify
import pandas as pd

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.text_emotion_model import predict_emotion

multimodal_routes = Blueprint("multimodal_routes", __name__)

# --------------------------------------------------
# SUGGESTIONS
# --------------------------------------------------
SUGGESTIONS = {
    "Healthy": {
        "mood": "Positive 😊",
        "messages": [
            "You seem emotionally balanced today. Keep your mind fresh and continue your healthy routine.",
            "Great! Your mood is positive. Stay active and keep spreading good vibes.",
            "You are doing well today. Maintain this energy with good food and proper rest.",
            "A positive mind is a powerful tool. Keep nurturing it with productive thoughts.",
            "Enjoy this good mood. Maybe try something creative or fun today!"
        ]
    },
    "Stable": {
        "mood": "Stable 🙂",
        "messages": [
            "Your mood looks stable. A short walk or light music can help maintain balance.",
            "You seem calm and steady. Keep your day simple and relaxed.",
            "This is a good time to focus on small productive tasks.",
            "Maintain this stability by staying hydrated and taking short breaks.",
            "A little mindfulness or stretching can keep you feeling steady."
        ]
    },
    "Low Mood Risk": {
        "mood": "Low Mood 😔",
        "messages": [
            "You seem a bit low today. Take rest and do something calming you enjoy.",
            "Try listening to your favorite music or watching something comforting.",
            "A short nap or warm drink might help lift your mood.",
            "Talk to a friend or spend some time with loved ones.",
            "Be gentle with yourself today. It’s okay to slow down."
        ]
    },
    "Stress Risk": {
        "mood": "Stressed 😣",
        "messages": [
            "You appear stressed. Take short breaks, relax your mind, and avoid overthinking.",
            "Step away from work for a few minutes and breathe deeply.",
            "Try organizing your tasks one by one instead of multitasking.",
            "A quick walk outside can reduce stress significantly.",
            "Drink water, stretch, and reset your mind before continuing."
        ]
    },
    "Anxiety Risk": {
        "mood": "Anxious 😟",
        "messages": [
            "Signs of anxiety detected. Slow breathing and talking to someone trusted may help.",
            "Focus on your breath for a minute. Inhale slowly, exhale gently.",
            "Ground yourself by noticing things around you.",
            "Avoid caffeine for a while and sit in a quiet place.",
            "Remind yourself that this feeling will pass."
        ]
    },
    "Emotional Discomfort": {
        "mood": "Disturbed 😞",
        "messages": [
            "You may be emotionally uncomfortable. Please prioritize rest and self-care.",
            "Give yourself permission to pause and relax today.",
            "Journaling your thoughts may help you feel lighter.",
            "Spend time in a calm environment or with someone supportive.",
            "Remember, it’s okay to seek help when you need it."
        ]
    }
}

# --------------------------------------------------
# SUBMIT DAILY TEXT
# --------------------------------------------------
@multimodal_routes.route("/submit-daily", methods=["POST"])
def submit_daily():
    text = request.form.get("text")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    result = predict_emotion(text)

    def polarity_score(emotions):
        score = 0
        for emo, pct in emotions.items():
            if emo in ["joy", "surprise"]:
                score += pct
            elif emo in ["sadness", "anger", "fear", "disgust"]:
                score -= pct
        return round(score / 100, 3)

    fusion_score = polarity_score(result["emotion_percentages"])

    DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
    os.makedirs(DATASET_DIR, exist_ok=True)
    log_path = os.path.join(DATASET_DIR, "multimodal_daily_logs.csv")

    row = {
        "date": str(datetime.date.today()),
        "text_score": fusion_score,
        "fusion_score": fusion_score,
        "dominant_emotion": result["dominant_emotion"],
        "mental_state": result["mental_state"]
    }

    pd.DataFrame([row]).to_csv(
        log_path,
        mode="a",
        header=not os.path.exists(log_path),
        index=False
    )

    # ✅ Random suggestion
    state = result["mental_state"]
    suggestion = SUGGESTIONS.get(
        state,
        {"mood": "Neutral 😐", "messages": ["Take care of yourself today."]}
    )

    mood = suggestion["mood"]
    msg = random.choice(suggestion["messages"])

    return jsonify({
        "date": row["date"],
        "score": fusion_score,
        "mental_state": state,
        "mood": mood,
        "message": msg
    })

# --------------------------------------------------
# WEEKLY REPORT
# --------------------------------------------------
@multimodal_routes.route("/weekly-report", methods=["GET"])
def weekly_report():
    path = os.path.join(PROJECT_ROOT, "datasets", "multimodal_daily_logs.csv")

    if not os.path.exists(path):
        return jsonify({
            "average_score": 0,
            "state": "No Data",
            "mood": "Neutral 😐",
            "message": "Not enough data yet. Add daily entries."
        })

    df = pd.read_csv(path, on_bad_lines="skip")
    df["fusion_score"] = pd.to_numeric(df["fusion_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fusion_score"])

    last7 = df.groupby(df["date"].dt.date)["fusion_score"].mean().tail(7)
    avg_score = round(last7.mean(), 3)

    # ✅ Decide state only
    if avg_score > 0.3:
        state = "Healthy"
    elif avg_score > 0:
        state = "Stable"
    elif avg_score > -0.5:
        state = "Low Mood Risk"
    else:
        state = "Stress Risk"

    suggestion = SUGGESTIONS[state]
    mood = suggestion["mood"]
    msg = random.choice(suggestion["messages"])

    return jsonify({
        "average_score": avg_score,
        "state": state,
        "mood": mood,
        "message": msg
    })

# --------------------------------------------------
# DRIFT DATA (LAST 7 DAYS)
# --------------------------------------------------
@multimodal_routes.route("/drift-data", methods=["GET"])
def drift_data():
    path = os.path.join(PROJECT_ROOT, "datasets", "multimodal_daily_logs.csv")

    if not os.path.exists(path):
        return jsonify({"labels": [], "scores": []})

    df = pd.read_csv(path, on_bad_lines="skip")
    df["fusion_score"] = pd.to_numeric(df["fusion_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fusion_score"])

    daily_avg = (
        df.groupby(df["date"].dt.date)["fusion_score"]
        .mean()
        .reset_index()
        .sort_values("date")
        .tail(7)
    )

    return jsonify({
        "labels": daily_avg["date"].astype(str).tolist(),
        "scores": daily_avg["fusion_score"].round(3).tolist()
    })

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from flask import send_file
import io
import datetime
import pandas as pd
import os

@multimodal_routes.route("/weekly-report-pdf")
def weekly_report_pdf():

    DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "multimodal_daily_logs.csv")

    if not os.path.exists(DATASET_PATH):
        return {"error": "No data available"}, 404

    df = pd.read_csv(DATASET_PATH, on_bad_lines="skip")
    df["fusion_score"] = pd.to_numeric(df["fusion_score"], errors="coerce")
    df = df.dropna(subset=["fusion_score"])

    last7 = df.tail(7)
    avg_score = round(last7["fusion_score"].mean(), 3)

    if avg_score > 0.3:
        state = "Healthy"
        mood = "Positive 🙂"
        note = "Positive emotional well-being observed."
    elif avg_score > 0:
        state = "Stable"
        mood = "Stable 🙂"
        note = "Emotionally stable with minor fluctuations."
    else:
        state = "Stress Risk"
        mood = "Stressed 😣"
        note = "Elevated stress indicators detected."

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    def line(text):
        nonlocal y
        pdf.drawString(40, y, text)
        y -= 18

    # HEADER
    pdf.setFont("Helvetica-Bold", 14)
    line("MENTAL HEALTH ASSESSMENT REPORT")
    y -= 10

    pdf.setFont("Helvetica", 10)
    line("Patient ID       : Anonymous User")
    line("Assessment Period: Last 7 Days")
    line(f"Report Date      : {datetime.date.today()}")
    line("System           : MINDTRAC-AI")

    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    line("1. CLINICAL SUMMARY")

    pdf.setFont("Helvetica", 10)
    line("AI-based emotional analysis was conducted using")
    line("sentiment polarity and emotion drift detection.")

    y -= 15
    pdf.setFont("Helvetica-Bold", 12)
    line("2. WEEKLY EMOTIONAL SCORE")

    pdf.setFont("Helvetica", 10)
    line(f"Average Emotional Score: {avg_score}")

    y -= 15
    pdf.setFont("Helvetica-Bold", 12)
    line("3. DOMINANT MENTAL STATE")

    pdf.setFont("Helvetica", 10)
    line(f"Mental State : {state}")
    line(f"Mood         : {mood}")

    y -= 15
    pdf.setFont("Helvetica-Bold", 12)
    line("4. OBSERVATIONS")

    pdf.setFont("Helvetica", 10)
    line(note)

    y -= 15
    pdf.setFont("Helvetica-Bold", 12)
    line("5. RECOMMENDATIONS")

    pdf.setFont("Helvetica", 10)
    line("• Maintain balanced daily routine")
    line("• Adequate rest and hydration")
    line("• Mindfulness or relaxation activities")

    y -= 15
    pdf.setFont("Helvetica-Bold", 12)
    line("6. DISCLAIMER")

    pdf.setFont("Helvetica", 9)
    line("This report is AI-generated and not a medical diagnosis.")
    line("Consult a licensed professional if needed.")

    y -= 30
    pdf.setFont("Helvetica-Oblique", 10)
    line("Doctor / System Signature:")
    line("MINDTRACE-AI Clinical Engine")

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="Weekly_Mental_Health_Report.pdf",
        mimetype="application/pdf"
    )
