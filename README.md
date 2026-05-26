# Live Translation with Gemini 3.1 Flash Lite

A real-time speech translation web app that:

1. Streams microphone audio (16 kHz int16 PCM) from the browser to a
   FastAPI backend over a single persistent WebSocket — same model as
   the [Gemini Live API Native Audio reference project][ref].
2. Runs **[DeepFilterNet v3][df]** on every incoming chunk as an
   *acoustic VAD* (denoise + RMS gate). When **two short pauses are
   detected within a short window**, the buffered utterance is flushed.
3. Sends the flushed WAV to **Gemini 3.1 Flash Lite** (Vertex AI, region
   configurable including `global`) and gets back JSON containing the
   verbatim **input transcription** and the **output translation**.
4. Optionally sends the translation to a **Gemini TTS** model and
   streams the synthesized audio back to the browser for playback.

[ref]: https://github.com/jerryscy/Live-translation-with-Gemini-Live-API-Native-Audio
[df]:  https://github.com/Rikorose/DeepFilterNet

---

## Architecture

```
┌──────────────┐  16 kHz PCM (binary)  ┌──────────────────────────┐
│   Browser    │ ───────────────────▶ │   FastAPI / WebSocket    │
│ AudioWorklet │                       │                          │
│              │ ◀──────────────────── │  TranslationSession      │
└──────────────┘  JSON events +        │   ├─ DeepFilterNet v3    │
   ▲              24 kHz PCM (TTS)     │   │   (denoise + VAD)    │
   │ playback (WAV)                    │   ├─ Pause aggregator    │
                                        │   │   (2 pauses → flush) │
                                        │   ├─ Gemini 3.1 Flash    │
                                        │   │   Lite (audio in →   │
                                        │   │   JSON {transcript,  │
                                        │   │         translation})│
                                        │   └─ Gemini TTS (opt.)   │
                                        └──────────────────────────┘
```

* **Frontend** (`static/index.html`, `static/audio-processor.js`):
  vanilla JS, AudioWorklet for low-latency PCM capture, single WebSocket.
* **Backend** (`main.py`, `translator.py`): FastAPI + `google-genai` SDK
  on Vertex AI mode.
* **VAD**: DeepFilterNet v3 denoising followed by a per-frame RMS gate;
  two pause edges inside a short window trigger the translation. Falls
  back gracefully to a raw-audio RMS gate if DeepFilterNet can't be
  loaded.

The "two pauses in a short period" heuristic was chosen because a single
brief pause is often just a hesitation. Waiting for the *second* pause
gives the speaker a chance to continue a thought before the system
commits the utterance to translation, while still keeping latency low.

---

## Configurable from the UI

All of the following can be changed at runtime from the right-hand
sidebar and are pushed to the backend with an `update_config` message:

| Setting | Notes |
|---|---|
| **Google Cloud project** | If your `.env` doesn't set one, type it here. |
| **Gemini 3.1 region** | Includes `global` plus all standard Vertex AI regions. |
| **Input language** | BCP-47 code passed to the translator prompt. |
| **Output language** | BCP-47 code passed to the translator prompt. |
| **Generate audio (TTS)** | Toggle to enable / disable Gemini TTS playback. |
| **TTS region** | Independent because TTS is not yet available everywhere. |
| **TTS voice** | Prebuilt Gemini voices (Kore, Puck, …). |

---

## Prerequisites

* Python 3.10+
* Google Cloud SDK (`gcloud`)
* A Google Cloud project with the **Vertex AI API** enabled
* *(Optional)* The Rust toolchain — only required if you want to build
  DeepFilterNet's `deepfilterlib` from source on macOS Apple Silicon
  (PyPI has prebuilt wheels for Linux x86_64 / Windows; arm64 macOS
  currently has to compile it). Install with `brew install rust` or
  via <https://rustup.rs/>.

---

## Install

### 1. Core dependencies (always required)

```bash
git clone <this-repo>
cd Live-translation-with-Gemini-3.1-Flash-Lite

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

This installs FastAPI, uvicorn, the `google-genai` SDK, and a few
utilities — enough to run the full pipeline using a **fallback
raw-audio RMS VAD** (no extra system deps needed).

### 2. (Optional) DeepFilterNet v3 noise-suppression VAD

For the noise-robust DeepFilterNet v3 acoustic VAD, install the extra
stack listed in `requirements-vad.txt`:

```bash
# macOS arm64 only: install Rust first (skip on Linux/Windows x86_64)
brew install rust          # or: curl https://sh.rustup.rs -sSf | sh

pip install -r requirements-vad.txt
```

The `DeepFilterNet3` checkpoint (~10 MB) is auto-downloaded into
`~/.cache/DeepFilterNet/` on first run.

> If DeepFilterNet isn't installed (or fails to load for any reason)
> the backend automatically falls back to the raw-audio RMS VAD — you
> still get a fully working app, just without DeepFilterNet's noise
> suppression.

#### Runtime tuning

Two environment variables control DFN3 at runtime (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DEEPFILTERNET_ENABLED` | `1` | Set to `0` / `false` to skip loading DFN3 entirely and fall back to the RMS-only VAD. |
| `DEEPFILTERNET_ATTEN_LIM_DB` | `30` | Caps how aggressively DFN3 attenuates noise. Lower values (15–30 dB) leave a little residual ambient noise but sound far more natural and usually improve ASR / Gemini transcription accuracy; higher values (60–100 dB) maximise suppression at the cost of occasional speech artefacts. |

Internally the backend uses a **streaming wrapper** (`DFStreamingDenoiser`)
that keeps a 160 ms rolling context window between WebSocket chunks so
DFN3 sees a continuous signal rather than independent slices — this
eliminates the periodic clicks/artefacts you would otherwise get at
chunk boundaries.


---

## Configure

```bash
cp .env.example .env
# Edit .env and set GOOGLE_CLOUD_PROJECT.
```

Authenticate Application Default Credentials (the backend will prompt
you to run this automatically if it's missing):

```bash
gcloud auth application-default login
```

If you hit SSL certificate issues on macOS:

```bash
export SSL_CERT_FILE=$(python3 -m certifi)
```

---

## Run

```bash
python3 main.py
```

Then open <http://127.0.0.1:8000>.

1. (Optional) Adjust language, region, and TTS settings in the sidebar
   and click **Apply Settings**.
2. Click **Start Recording** and speak.
3. Each time the VAD detects two short pauses, the system translates
   the buffered utterance, displays the source transcription + the
   translation in the chat, and (if TTS is enabled) plays back the
   translated audio.
4. Click **Stop** to stop streaming (this also force-flushes whatever
   audio is still in the buffer), or **Flush** to force a translation
   without stopping.

---

## File layout

```
.
├── main.py                  # FastAPI app, WebSocket endpoint
├── translator.py            # DeepFilterNet v3 VAD + Gemini calls
├── requirements.txt         # Core deps (always required)
├── requirements-vad.txt     # Optional DeepFilterNet v3 deps
├── .env.example
├── README.md
└── static/
    ├── index.html           # UI + WebSocket client
    └── audio-processor.js   # AudioWorklet: f32 → int16 PCM
```

---

## How "two pauses in a short period" is computed

In `translator.py → PauseDetector`:

* Audio is processed in 30 ms frames.
* A frame is "silent" if its RMS (on the *denoised* signal) is below
  `energy_threshold` (default `0.012`).
* A *pause edge* fires the instant a contiguous silent run reaches
  `min_pause_ms` (default 400 ms).
* When a pause edge fires and the previous pause edge was less than
  `pause_gap_ms` (default 2200 ms) ago, the current utterance is
  **flushed** to Gemini 3.1 Flash Lite for translation.
* A 20 s hard maximum (`max_utterance_ms`) prevents runaway buffers.

Tweak these constants at the top of `PauseDetector.__init__` to taste.

---

## Why a request-per-utterance model?

The original reference project uses the Gemini Live API, which is a
persistent streaming session. Gemini 3.1 Flash Lite is a standard
`generateContent` model and doesn't expose a live-streaming session, so
the natural design is:

* **Client**: persistent WebSocket, continuously streaming audio.
* **Server**: chunks the stream into utterances using DeepFilterNet v3 VAD
  and fires one `generateContent` request per utterance.

This keeps the perceived latency low (we only translate complete
utterances), avoids paying for long idle sessions, and is much easier
to deploy across multiple regions / models because each request is
independent.
