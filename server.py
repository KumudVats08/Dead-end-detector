"""HTTP layer for the custom frontend. Run: uvicorn server:app --reload"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from pipeline import FIELD_NAMES, run_pipeline

ROOT = Path(__file__).parent
app = FastAPI(title="Dead-End Detector")
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    idea: str
    demo: bool = False


def _encode(value):
    # Ranking scores may be numpy floats, which json cannot serialize by default.
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    events: queue.Queue = queue.Queue()
    idea = request.idea.strip()

    def step(name: str) -> None:
        events.put({"type": "step", "step": name})

    def work() -> None:
        try:
            try:
                result = run_pipeline(idea, progress_callback=step, use_demo=request.demo)
            except Exception as error:  # live run failed: fall back to saved demo data
                events.put({"type": "notice", "message": f"Live research stopped: {error}"})
                result = run_pipeline(idea, progress_callback=step, use_demo=True)
            if not result.get("demo") and os.environ.get("SAVE_DEMO") == "1":
                today = datetime.now()
                saved_result = {
                    "idea": idea,
                    "saved": f"{today:%b} {today.day}, {today:%Y}",
                    "result": result,
                    "fields": list(FIELD_NAMES),
                }
                try:
                    (ROOT / "static" / "demo_result.json").write_text(
                        json.dumps(saved_result, default=_encode, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    logger.exception("Unable to save the latest live result")
            events.put({"type": "result", "result": result, "fields": list(FIELD_NAMES)})
        except Exception as error:
            events.put({"type": "error", "message": str(error)})
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            yield json.dumps(item, default=_encode) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
