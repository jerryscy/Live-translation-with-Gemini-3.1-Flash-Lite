"""FastAPI entry point.

Browser  <--WebSocket-->  FastAPI  <-->  Gemini 3.1 Flash Lite / Gemini TTS

Audio flow:
  * Client streams 16 kHz int16 PCM frames over the WebSocket (binary).
  * Backend feeds them to `TranslationSession` which performs DeepFilterNet
    denoising + double-pause VAD. On every utterance flush it calls the
    Gemini 3.1 Flash Lite model (audio in → JSON {transcription, translation}
    out) and optionally Gemini TTS.
  * Backend pushes:
       - JSON text messages (transcription / translation / status events)
       - binary frames containing raw int16 PCM @ 24 kHz mono (TTS audio)
    back to the client.

Config:
  * `GOOGLE_CLOUD_PROJECT` must be set (env or .env).
  * Default Gemini region is `global`; can be changed per-session from the UI.
  * Authentication uses gcloud Application Default Credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from translator import TranslationConfig, TranslationSession

load_dotenv()


# ---------------------------------------------------------------------- gcloud
def check_gcloud_auth() -> None:
    """Make sure Application Default Credentials are available."""
    try:
        subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"],
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No ADC found — running `gcloud auth application-default login`…")
        subprocess.run(["gcloud", "auth", "application-default", "login"])


# ------------------------------------------------------------------------ app
app = FastAPI(title="Live Translation with Gemini 3.1 Flash Lite")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def _on_startup() -> None:
    check_gcloud_auth()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        print(
            "WARNING: GOOGLE_CLOUD_PROJECT is not set. Set it in a .env "
            "file or as an env var before connecting a client."
        )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ------------------------------------------------------------------- websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    print("[ws] Client connected")

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    cfg = TranslationConfig(
        project_id=project_id,
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        translation_model=os.getenv("TRANSLATION_MODEL", "gemini-3.1-flash-lite"),
        tts_model=os.getenv("TTS_MODEL", "gemini-3.1-flash-tts-preview"),
        # gemini-3.1-flash-tts-preview is only served from the global endpoint.
        tts_location=os.getenv("TTS_LOCATION", "global"),
    )

    # Single asyncio.Lock around websocket sends so concurrent translation
    # tasks can't interleave a binary frame in the middle of a text frame.
    send_lock = asyncio.Lock()

    async def send_event(event: dict) -> None:
        try:
            async with send_lock:
                await websocket.send_text(json.dumps(event))
        except Exception as exc:
            print(f"[ws] send_event failed: {exc}")

    async def send_audio(pcm_bytes: bytes) -> None:
        try:
            async with send_lock:
                # Prefix with a tiny JSON envelope so the client can tell
                # what to do with the upcoming binary frame.
                await websocket.send_text(json.dumps({"type": "audio_header",
                                                      "sample_rate": 24000,
                                                      "encoding": "pcm_s16le"}))
                await websocket.send_bytes(pcm_bytes)
        except Exception as exc:
            print(f"[ws] send_audio failed: {exc}")

    session = TranslationSession(cfg, send_event=send_event, send_audio=send_audio)

    # Hello: tell client the active config so the UI can reflect it.
    await send_event({
        "type": "hello",
        "config": _config_to_dict(cfg),
        "project_set": bool(project_id),
    })

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                if not cfg.project_id:
                    await send_event({"type": "error",
                                      "message": "GOOGLE_CLOUD_PROJECT is not configured."})
                    continue
                await session.add_audio(msg["bytes"])

            elif "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                await _handle_control(data, session, cfg, send_event)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[ws] error: {exc}")
    finally:
        await session.shutdown()
        print("[ws] Client disconnected")


# ------------------------------------------------------------------- helpers
def _config_to_dict(cfg: TranslationConfig) -> dict:
    return {
        "project_id": cfg.project_id,
        "location": cfg.location,
        "translation_model": cfg.translation_model,
        "tts_model": cfg.tts_model,
        "tts_location": cfg.tts_location,
        "tts_voice": cfg.tts_voice,
        "tts_enabled": cfg.tts_enabled,
        "source_language": cfg.source_language,
        "source_language_code": cfg.source_language_code,
        "target_language": cfg.target_language,
        "target_language_code": cfg.target_language_code,
    }


async def _handle_control(
    data: dict,
    session: TranslationSession,
    cfg: TranslationConfig,
    send_event,
) -> None:
    action = data.get("action")

    if action == "update_config":
        session.update_config(
            location=data.get("location"),
            translation_model=data.get("translation_model"),
            tts_enabled=data.get("tts_enabled"),
            tts_voice=data.get("tts_voice"),
            tts_location=data.get("tts_location"),
            source_language=data.get("source_language"),
            source_language_code=data.get("source_language_code"),
            target_language=data.get("target_language"),
            target_language_code=data.get("target_language_code"),
            project_id=data.get("project_id"),
        )
        await send_event({"type": "config_updated", "config": _config_to_dict(cfg)})

    elif action == "flush":
        await session.manual_flush()

    elif action == "ping":
        await send_event({"type": "pong"})


# ------------------------------------------------------------------- __main__
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
