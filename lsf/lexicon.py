"""The LSF vocabulary catalogue.

This is the *catalogue* of signs the app knows about (curated beginner LSF
vocabulary, grouped by theme). It is separate from what has actually been
*trained*: a catalogue word becomes recognisable once you record at least one
sample of it from the web UI (see `recognizer.SignRecognizer`).

Users can extend the catalogue at runtime (`add_word`); additions are persisted
back to the same JSON file.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Optional

# Authoritative LSF video dictionary (one signed video per word).
ELIX_BASE = "https://dico.elix-lsf.fr/dictionnaire/"


def _slug_to_gloss(text: str) -> str:
    """Normalise free text into an LSF-style gloss (uppercase French)."""
    return re.sub(r"\s+", " ", text).strip().upper()


def elix_url(fr: str) -> str:
    """Build the Elix dictionary URL for a French word (accents stripped, slugged)."""
    s = unicodedata.normalize("NFKD", (fr or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", " ").replace("’", " ")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return ELIX_BASE + s


class Lexicon:
    def __init__(self, path: str = "signs/lexicon.json") -> None:
        self.path = path
        self.meta: dict = {}
        self.words: list[dict] = []
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.meta, self.words = {}, []
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.meta = data.get("meta", {})
        self.words = data.get("words", [])

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"meta": self.meta, "words": self.words}, fh,
                          ensure_ascii=False, indent=2)
        except OSError:
            pass  # ephemeral/read-only FS: keep additions in memory only

    @property
    def glosses(self) -> list[str]:
        return [w["gloss"] for w in self.words]

    def categories(self) -> list[str]:
        seen: list[str] = []
        for w in self.words:
            cat = w.get("category", "Autres")
            if cat not in seen:
                seen.append(cat)
        return seen

    def add_word(
        self,
        fr: str,
        *,
        en: str = "",
        category: str = "Personnalisés",
        tip: str = "",
        gloss: Optional[str] = None,
    ) -> dict:
        """Add (or return existing) catalogue entry for a French word."""
        gloss = _slug_to_gloss(gloss or fr)
        for w in self.words:
            if w["gloss"] == gloss:
                return w
        entry = {"gloss": gloss, "fr": fr.strip(), "en": en, "category": category, "tip": tip}
        self.words.append(entry)
        self.save()
        return entry

    def as_entries(self, trained_counts: Optional[dict] = None) -> list[dict]:
        """Catalogue merged with training state, for the UI.

        Any trained gloss not in the catalogue is appended as a custom entry so
        nothing the recogniser knows is hidden from the browser.
        """
        trained_counts = trained_counts or {}
        out: list[dict] = []
        known = set()
        for w in self.words:
            g = w["gloss"]
            known.add(g)
            out.append(
                {**w, "samples": int(trained_counts.get(g, 0)),
                 "ref": elix_url(w.get("fr") or g)}
            )
        for gloss, count in trained_counts.items():
            if gloss not in known:
                out.append(
                    {
                        "gloss": gloss,
                        "fr": gloss.capitalize(),
                        "en": "",
                        "category": "Personnalisés",
                        "tip": "",
                        "samples": int(count),
                        "ref": elix_url(gloss),
                    }
                )
        return out
