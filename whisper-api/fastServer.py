from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
import os
import logging
import sys
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv

# ---------------- INIT ----------------
load_dotenv()
app = FastAPI()

logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s] %(asctime)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------- MODEL (CPU) ----------------
# int8 is fastest + lowest memory on CPU
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# ---------------- EMAIL ALERT ----------------
def send_email_alert(filename: str, duration: float):
    msg = EmailMessage()
    msg["Subject"] = "🎤 Whisper Transcription Used"
    msg["From"] = os.environ["ALERT_EMAIL"]
    msg["To"] = os.environ["ALERT_EMAIL_TO"]

    msg.set_content(f"""
A transcription job was submitted.

File: {filename}
Processing time: {duration:.2f} seconds
Timestamp: {datetime.utcnow()} UTC
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(
            os.environ["ALERT_EMAIL"],
            os.environ["ALERT_EMAIL_PASSWORD"]
        )
        server.send_message(msg)

# ---------------- ENDPOINT ----------------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    start_time = time.perf_counter()
    logger.info(f"Job Started: {file.filename}")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # -------- faster-whisper transcription --------
    segments, info = model.transcribe(
        tmp_path,
        language="en",
        beam_size=1,
        vad_filter=True
    )

    # Match old Whisper response format
    text = "".join(segment.text for segment in segments)

    # Timing + alert
    job_time = time.perf_counter() - start_time
    send_email_alert(file.filename, job_time)

    logger.info(f"Job {file.filename} completed in {job_time:.2f}s")

    os.unlink(tmp_path)

    return {"text": text}

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
