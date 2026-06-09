#!/usr/bin/env python3
"""Train the recogniser from Elix LSF dictionary videos.

For each word in the catalogue (signs/lexicon.json) this fetches the word's
reference sign video from the Elix dictionary (dico.elix-lsf.fr), runs it
through the *same* pipeline as the live recorder, and stores a DTW template in
signs/lsf_signs.json. The catalogue thus becomes recognisable without recording
each sign by hand.

Usage:
    python tools/elix_train.py                 # all catalogue words
    python tools/elix_train.py --limit 5       # first 5 (quick test)
    python tools/elix_train.py --words BONJOUR MERCI
    python tools/elix_train.py --cache clips/elix   # keep downloaded mp4s

Politeness / usage notes:
    * Elix robots.txt permits crawling (empty Disallow); we still rate-limit
      (--delay, default 1s) and send a descriptive User-Agent.
    * The videos are © Signes de sens (Elix). This derives abstract landmark
      templates for personal/on-device use and does NOT redistribute the videos.
      Respect Elix's terms — don't redistribute their video files.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsf.landmarks import HolisticTracker  # noqa: E402
from lsf.lexicon import Lexicon  # noqa: E402
from lsf.recognizer import SignRecognizer  # noqa: E402
from tools.import_videos import import_clip  # noqa: E402

UA = "LSF-transcripter/0.1 (educational; +https://github.com/RebornX10/LSF_transcription)"
ELIX_PAGE = "https://dico.elix-lsf.fr/dictionnaire/"
# Primary: the <video src> on a word page; fallback: any Elix mp4 on the page.
VIDEO_RE = re.compile(r"""<video[^>]*\bsrc=["'](https?://[^"']+?\.mp4)["']""", re.I)
ANY_MP4_RE = re.compile(r"""https?://[^"'\s]+?\.mp4""", re.I)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def find_video_urls(html: str, limit: int) -> list[str]:
    """Distinct sign-video URLs on a page, in order, re-encode copies collapsed.

    Elix serves several files per word: the sign(s), plus `-2`/`-3`/`-encoded`
    re-encodes of the same clip. We group by the base filename (dropping those
    trailing markers) so each entry is a genuinely different rendition.
    """
    urls = VIDEO_RE.findall(html) or ANY_MP4_RE.findall(html)
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        base = re.sub(r"(-encoded)?(-\d+)?\.mp4$", "", u.rsplit("/", 1)[-1])
        if base in seen:
            continue
        seen.add(base)
        out.append(u)
        if len(out) >= limit:
            break
    return out


def _slug(fr: str, *, keep_accents: bool, apostrophe: str) -> str:
    s = fr.strip().lower()
    if not keep_accents:
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.replace("'", apostrophe).replace("’", apostrophe)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def candidate_urls(fr: str):
    """Elix page URL candidates. Elix keeps accents in the slug, so the accented
    form is tried first, then apostrophe-stripped and de-accented fallbacks."""
    seen = set()
    for keep, apos in ((True, "-"), (True, ""), (False, "-")):
        slug = _slug(fr, keep_accents=keep, apostrophe=apos)
        if slug and slug not in seen:
            seen.add(slug)
            yield ELIX_PAGE + urllib.parse.quote(slug)


def cache_name(gloss: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", gloss.lower()).strip("_") or "sign"


def main() -> int:
    p = argparse.ArgumentParser(description="Train templates from Elix sign videos.")
    p.add_argument("--lexicon", default="signs/lexicon.json")
    p.add_argument("--templates", default="signs/lsf_signs.json")
    p.add_argument("--cache", default=None, help="Dir to keep downloaded mp4s (else temp).")
    p.add_argument("--limit", type=int, default=0, help="Only the first N words.")
    p.add_argument("--words", nargs="*", help="Only these glosses.")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between requests.")
    p.add_argument("--count", type=int, default=1,
                   help="Distinct videos to train per word (= samples added).")
    p.add_argument("--skip", type=int, default=0,
                   help="Skip the first N videos (e.g. --skip 1 --count 1 adds only the 2nd).")
    args = p.parse_args()

    lex = Lexicon(args.lexicon)
    words = lex.words
    if args.words:
        wanted = {w.upper() for w in args.words}
        words = [w for w in words if w["gloss"].upper() in wanted]
    if args.limit:
        words = words[: args.limit]
    if not words:
        print("No matching words.")
        return 1

    cache = args.cache or tempfile.mkdtemp(prefix="elix_")
    os.makedirs(cache, exist_ok=True)

    recognizer = SignRecognizer(templates_path=args.templates, use_heuristics=False)
    tracker = HolisticTracker()

    samples = 0
    added_words = 0
    skipped: list[str] = []
    try:
        for i, w in enumerate(words, 1):
            gloss, fr = w["gloss"], w.get("fr", w["gloss"])
            print(f"[{i}/{len(words)}] {gloss}")
            urls: list[str] = []
            for page in candidate_urls(fr):
                try:
                    html = fetch(page).decode("utf-8", "ignore")
                except Exception as exc:
                    print(f"  ! fetch failed ({page}): {exc}")
                    continue
                urls = find_video_urls(html, args.skip + args.count)
                if urls:
                    break
                time.sleep(0.3)  # politeness between candidate tries

            selected = urls[args.skip : args.skip + args.count]
            if not selected:
                print("  ! no (additional) sign video found — skipped")
                skipped.append(gloss)
                continue

            got = 0
            for idx, video_url in enumerate(selected, start=args.skip):
                path = os.path.join(cache, f"{cache_name(gloss)}_{idx}.mp4")
                if not os.path.exists(path):
                    try:
                        data = fetch(video_url)
                        with open(path, "wb") as fh:
                            fh.write(data)
                        print(f"  ↓ {os.path.basename(video_url)} ({len(data)//1024} KB)")
                    except Exception as exc:
                        print(f"  ! download failed: {exc}")
                        continue
                if import_clip(tracker, recognizer, gloss, path):
                    samples += 1
                    got += 1
                time.sleep(args.delay)

            if got:
                added_words += 1
            else:
                skipped.append(gloss)
    finally:
        tracker.close()

    print(f"\nAdded {samples} sample(s) across {added_words}/{len(words)} words -> {args.templates}.")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")
    print(f"Videos cached in: {cache}")
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
