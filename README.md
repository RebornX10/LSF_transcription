---
title: LSF Interpreter
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

DISCLAIMER: Used claude for django, UI, Gunicorn and some comments (and maybe git)

# LSF_transcripter — French Sign Language interpreter

Real-time **Langue des Signes Française (LSF)** interpreter. It reads the signer
from a camera, tracks the body parts that carry meaning in sign language —
**fingers / hand shape, eyes & gaze, mouth, chest and arms** — draws the
skeleton over the live feed, and transcribes recognised signs underneath.

A **Django** web UI shows the live camera feed with the skeleton at the top and
the running transcription at the bottom (HTML / CSS / JS in separate files).

**Architecture:** the **browser** captures the camera and posts frames; the
**server** runs MediaPipe + the recognizer and returns landmarks + glosses that
the browser draws. The same app therefore runs locally *and* on Hugging Face
(where the server has no webcam), and dual-pixel detection reads the real client
device. See [Deploy](#deploy--docker--hugging-face).

---

## How it works

```
camera frame
   │
   ├─▶ MediaPipe Holistic ──▶ pose (33) · face mesh (468) · 2×hands (21)
   │         │                     fingers ▸ eyes ▸ mouth ▸ chest ▸ arms
   │         ▼
   │   depth refinement  ◀── dual-pixel / depth stream  (or MiDaS, or none)
   │         ▼
   │   normalised body-relative features  (features.py)
   │         ▼
   │   sign recogniser  ──▶  DTW templates · trained model · heuristics
   │         ▼
   └─▶ skeleton overlay  +  transcribed gloss
```

| Body part the brief asked for | Where it comes from              |
|-------------------------------|----------------------------------|
| Fingers / hand shape          | `left_hand` / `right_hand` (21 pts each) |
| Eyes / gaze                   | face-mesh eye + iris landmarks   |
| Mouth (mouthing, non-manuals) | face-mesh lip landmarks          |
| Chest                         | pose shoulders + hips            |
| Arms                          | pose shoulders / elbows / wrists |
| Hand                          | hand wrist + pose wrist          |

### Dual-pixel sensor detection (e.g. Pixel devices)

"Dual-pixel" depth is produced by the *camera stack* on certain devices (e.g.
Pixel phones); it lives on the **client device**, not the server. Detection
therefore runs in the browser ([app.js](web/interpreter/static/interpreter/app.js),
`detectDualPixel`): it probes the `MediaStreamTrack` capabilities
(`getCapabilities`, `getSupportedConstraints`), enumerates devices for a depth
camera, and matches the camera label (`pixel`, `depth`, `tof`, `truedepth`). The
result is POSTed to `/depth`; the status pill then shows `dual_pixel` vs
`mediapipe`, with the reason in its tooltip.

Honest limit: browsers don't generally expose the *dual-pixel depth map* itself,
so when detected we flag the source and keep MediaPipe's relative `z`. If a
platform does hand you a depth map, feed it to
`DepthEstimator.depth_for_frame(rgb, external_depth=...)` and every landmark's
`z` is resampled from it. A server-side MiDaS monocular fallback is also
available (`enable_monocular_depth`, needs `torch`).

### Recognition — be realistic

There is no off-the-shelf production LSF translation model. This project gives
you a working, extensible pipeline rather than a magic black box:

0. **Vocabulary catalogue.** It ships with ~80 common LSF words
   (`signs/lexicon.json`) grouped by theme — greetings, family, verbs,
   questions, emotions, time, food, places, colours. These are the *targets*
   shown in the web UI's vocabulary browser; a word becomes *recognisable* once
   you train it (step 1). You can add your own words from the UI too.
1. **Learn-by-example (default).** Click **Entraîner** next to a word, perform
   the sign, click **Sauvegarder**. Samples are stored as normalised landmark
   sequences and matched with **Dynamic Time Warping**. Runs on a laptop CPU, no
   training needed, good for a small/medium vocabulary.
2. **Trained model (optional).** Drop a Keras `models/model.h5` (+
   `models/model.labels.json`) and it's used automatically. Train your own
   sequence model on a corpus for real coverage.
3. **Heuristics.** A couple of geometry-only signs so the demo isn't silent
   before you've recorded anything.

---

## Setup

```bash
./setup.sh                       # creates .venv and installs deps
source .venv/bin/activate
```

Requires Python 3.9–3.11 (MediaPipe constraint). On macOS, grant camera access
to your terminal in *System Settings ▸ Privacy & Security ▸ Camera*.

## Run

**Quick start — one command (creates the venv on first run):**
```bash
./run.sh            # → http://127.0.0.1:8000   (ships pre-trained, 72 LSF signs)
```

**Web UI (Django) — everything is in the browser, no terminal needed:**
```bash
python web/manage.py runserver
# open http://127.0.0.1:8000  (allow the camera when prompted)
```
The browser captures the camera; `getUserMedia` needs a secure context, which
`localhost` satisfies (on a LAN IP it won't — use localhost or HTTPS).

* **Top-left:** live feed with the skeleton (drawn in the browser).
* **Bottom-left:** transcription.
* **Right:** vocabulary browser — search/filter the ~80-word catalogue, see which
  signs are trained, **Entraîner** a word (record → *Sauvegarder*), delete
  trained samples, **Ajouter** a custom word, or click a word for its tip + Elix
  sign video.
* Status pills show the depth source (incl. `dual_pixel` when detected), FPS, and
  how many signs are trained.

**Standalone window (local OpenCV, uses the server's own webcam):**
```bash
python main.py            # keys: r=record, s=save, c=clear, q=quit
```

## Deploy — Docker / Hugging Face

**Docker (works locally too — the camera is the browser's, no device passthrough):**
```bash
docker build -t lsf .
docker run -p 7860:7860 lsf
# open http://localhost:7860
```

**Hugging Face Spaces:** push the repo to a **Docker** Space (the front-matter at
the top of this README sets `sdk: docker` and `app_port: 7860`). HF builds the
[Dockerfile](Dockerfile) and serves it over HTTPS, so the browser camera and
dual-pixel detection work. Tuning is via env vars: `LSF_MODEL_COMPLEXITY`
(0–2), `LSF_REFINE_FACE` (0/1), `LSF_DEBUG`, `LSF_SECRET_KEY`.

> Note: a hosted Space has an **ephemeral, often read-only** filesystem — trained
> templates/added words persist in memory for the session but may not survive a
> restart. For durable training, run locally or import clips (below) into a
> mounted volume.

### Pre-training words from video clips (batch)

Rather than recording each sign live, you can batch-import reference clips so the
words are recognised immediately. Name clips by their gloss:

```
clips/BONJOUR.mp4            # one sample for BONJOUR
clips/AU REVOIR/01.mp4       # several samples for AU REVOIR
clips/AU REVOIR/02.mp4
```
```bash
python tools/import_videos.py clips/
# or a single file:
python tools/import_videos.py merci.mov --gloss MERCI
```
This runs the clips through the same pipeline as the live recorder and writes
templates into `signs/lsf_signs.json`. Record 2–3 clips per sign (slightly
different speed/framing) for better accuracy. Note: there is no public drop-in
*pre-trained LSF model* — reference clips (filmed by you or any LSF video you're
allowed to use) are how the vocabulary gets "pre-trained".

**Auto-train from the Elix dictionary.** [tools/elix_train.py](tools/elix_train.py)
fetches each catalogue word's reference sign video from
[dico.elix-lsf.fr](https://dico.elix-lsf.fr/) and trains a template from it —
**72 of the 81 words ship pre-trained** this way:
```bash
python tools/elix_train.py            # all words   (or --words BONJOUR MERCI …)
```
It rate-limits and identifies itself; robots.txt permits crawling. The Elix
videos are © Signes de sens — this derives abstract landmark templates for
personal/on-device use and does **not** redistribute the videos. The 9 unmatched
words are multi-word phrases (AU REVOIR, ÇA VA, …) — record those by hand.

---

## Layout

```
main.py                     standalone CLI runner (OpenCV window, local webcam)
tools/import_videos.py      batch clips → templates ("pre-train" words)
Dockerfile / .dockerignore  container build (HF Spaces / local)
requirements.txt            deps (mediapipe, opencv, django, gunicorn, whitenoise)
setup.sh                    venv bootstrap
signs/lexicon.json          ~80-word LSF vocabulary catalogue (themes + tips)
signs/lsf_signs.json        learned-sign templates (shared by CLI + web)
models/                     optional model.h5 + model.labels.json
lsf/
  landmarks.py              MediaPipe Holistic wrapper + compact points payload
  skeleton.py               server-side skeleton draw (CLI only)
  depth.py                  depth refinement + MiDaS fallback
  features.py               landmarks ▸ normalised body-relative feature vector
  recognizer.py             motion segmentation + DTW / model / heuristics
  lexicon.py                loads/extends the vocabulary catalogue (+ Elix links)
  camera.py                 InterpreterPipeline + local-webcam singleton (CLI)
  session.py                FrameSession: browser-pushed frames → landmarks+glosses
web/
  manage.py
  web/                      Django project (settings, urls, wsgi, asgi)
  interpreter/              Django app
    views.py                page · /process · /depth · vocab + training APIs
    urls.py
    templates/interpreter/index.html
    static/interpreter/style.css
    static/interpreter/app.js   getUserMedia · dual-pixel · /process · skeleton draw
```

## Limitations & next steps

* Out of the box it recognises only what you record (plus 2 rough heuristics).
  For broad LSF coverage, collect a dataset and train the model hook.
* LSF grammar (spatial reference, classifiers, facial grammar) is **not** parsed
  — output is a stream of glosses, not fluent French. A gloss→French language
  step would sit after the recogniser.
* Dual-pixel depth needs hardware that exposes it (see above).
