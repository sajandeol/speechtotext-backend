import whisper
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
import os
import logging
import sys
import time

app = FastAPI()

logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s] %(asctime)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
# CPU-safe Whisper (your original)
model = whisper.load_model("medium", device="cpu")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # save uploaded file to temporary file
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    start_time = time.perf_counter()
    logger.info("Job Started: "+file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # run whisper
    #result = model.transcribe(tmp_path, fp16=False)
    #transcribe using only English
    result = model.transcribe(
     tmp_path,
     fp16=False,
     language="en",
     task="transcribe"
    )

    end_time = time.perf_counter()
    jobTime = end_time - start_time
    logger.info(f"Job {file.filename} completed in {jobTime:.2f}s")
    # clean up
    os.unlink(tmp_path)

    return {"text": result["text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
