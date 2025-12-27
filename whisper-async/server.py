# server.py
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from tasks import celery_app, UPLOAD_DIR, RESULT_DIR

app = FastAPI()

@app.post("/transcribe")
async def submit_transcription(audio: UploadFile = File(...)):
    """
    Accept an upload, save it, enqueue a Celery task, return job_id.
    Field name expected: "audio"
    """
    # Save upload
    suffix = Path(audio.filename).suffix or ".wav"
    jobname = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{jobname}{suffix}"

    contents = await audio.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Enqueue Celery task
    result_path = str(RESULT_DIR / f"{jobname}.json")
    task = celery_app.send_task("tasks.transcribe_task", args=[str(save_path), result_path])

    return {"job_id": task.id}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    res = AsyncResult(job_id, app=celery_app)
    state = res.state
    info = res.info if hasattr(res, "info") else None

    # If succeeded, include result
    if state == "SUCCESS":
        result = res.result  # should contain dict with 'text' or 'result_path'
        return {"status": "completed", "result": result}
    elif state in ("PENDING","RECEIVED"):
        return {"status": "queued"}
    elif state in ("PROCESSING", "STARTED"):
        # Provide any meta info if available
        meta = info if isinstance(info, dict) else {}
        return {"status": "processing", "meta": meta}
    elif state == "FAILURE":
        return {"status": "error", "error": str(res.result)}
    else:
        return {"status": state, "info": info}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    res = AsyncResult(job_id, app=celery_app)
    if res.state != "SUCCESS":
        return JSONResponse(status_code=202, content={"status": res.state})
    # SUCCESS
    payload = res.result
    # If worker returned text directly
    if isinstance(payload, dict) and "text" in payload:
        return {"status": "completed", "text": payload["text"]}
    # If worker returned a result_path
    if isinstance(payload, dict) and "result_path" in payload:
        try:
            import json
            with open(payload["result_path"], "r", encoding="utf-8") as f:
                return {"status": "completed", **json.load(f)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"status": "completed", "result": payload}
