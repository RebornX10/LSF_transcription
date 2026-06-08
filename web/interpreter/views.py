"""Views for the browser-camera architecture.

The browser captures the webcam and POSTs frames to ``/process``; the server
runs MediaPipe + the DTW recognizer (via the ``FrameSession`` singleton) and
returns landmarks + recognised glosses, which the browser draws. This is what
lets the same Django app run locally *and* on Hugging Face (where the server
has no camera).
"""

from __future__ import annotations

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from lsf.lexicon import Lexicon
from lsf.session import FrameSession

_LEXICON: Lexicon | None = None
MAX_FRAME_BYTES = 4 * 1024 * 1024  # 4 MB guard for a posted frame


def _session() -> FrameSession:
    return FrameSession.instance(
        templates_path=settings.TEMPLATES_PATH,
        model_path=settings.MODEL_PATH,
        model_complexity=settings.MODEL_COMPLEXITY,
        refine_face_landmarks=settings.REFINE_FACE_LANDMARKS,
    )


def _lexicon() -> Lexicon:
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = Lexicon(settings.LEXICON_PATH)
    return _LEXICON


@require_GET
def index(request):
    return render(request, "interpreter/index.html")


@csrf_exempt
@require_POST
def process(request):
    """Receive one JPEG frame (raw body), return landmarks + any recognition."""
    body = request.body
    if not body:
        return JsonResponse({"ok": False, "error": "Empty frame."}, status=400)
    if len(body) > MAX_FRAME_BYTES:
        return JsonResponse({"ok": False, "error": "Frame too large."}, status=413)
    return JsonResponse(_session().process_jpeg(body))


@csrf_exempt
@require_POST
def depth(request):
    """Browser reports the device's depth/dual-pixel capability."""
    dp = (request.POST.get("dual_pixel") or "0") in ("1", "true", "True")
    _session().set_depth_capability(
        dual_pixel=dp,
        reason=(request.POST.get("reason") or "").strip(),
        label=(request.POST.get("label") or "").strip(),
    )
    return JsonResponse({"ok": True, "status": _session().status})


@require_GET
def status(request):
    return JsonResponse(_session().status)


# -- recording (learn-by-example) -------------------------------------------
@csrf_exempt
@require_POST
def record_start(request):
    label = (request.POST.get("label") or "").strip()
    if not label:
        return JsonResponse({"ok": False, "error": "Missing 'label'."}, status=400)
    _session().start_recording(label)
    return JsonResponse({"ok": True, "recording": label})


@csrf_exempt
@require_POST
def record_stop(request):
    saved = _session().stop_recording()
    return JsonResponse({"ok": saved is not None, "saved": saved})


@csrf_exempt
@require_POST
def record_cancel(request):
    _session().cancel_recording()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_POST
def clear(request):
    _session().clear_transcript()
    return JsonResponse({"ok": True})


# -- vocabulary -------------------------------------------------------------
@require_GET
def lexicon(request):
    lex = _lexicon()
    counts = _session().template_counts()
    return JsonResponse({"categories": lex.categories(), "words": lex.as_entries(counts)})


@csrf_exempt
@require_POST
def vocab_add(request):
    fr = (request.POST.get("fr") or "").strip()
    if not fr:
        return JsonResponse({"ok": False, "error": "Missing 'fr'."}, status=400)
    entry = _lexicon().add_word(
        fr,
        en=(request.POST.get("en") or "").strip(),
        category=(request.POST.get("category") or "Personnalisés").strip(),
        tip=(request.POST.get("tip") or "").strip(),
    )
    return JsonResponse({"ok": True, "word": entry})


@csrf_exempt
@require_POST
def vocab_delete(request):
    gloss = (request.POST.get("gloss") or "").strip()
    if not gloss:
        return JsonResponse({"ok": False, "error": "Missing 'gloss'."}, status=400)
    return JsonResponse({"ok": _session().delete_gloss(gloss), "gloss": gloss})
