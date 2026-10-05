"""Input-normalization defense for the guardrail.

The red-team results show the detector fails on attacks that hide the payload from its tokenizer
(invisible Unicode) or wrap it so the model can't read it (base64). Both are pre-classification
problems: if we normalize the input so the payload becomes visible plain text, the existing
detector should catch it. This module does that normalization; redteam_eval.py --defend applies it
and we measure the drop in bypass rate, which is the evidence that the diagnosis is correct.

Normalizations (all reversible-safe, applied before scoring only):
  * strip invisible/format characters: Unicode Tags block (U+E0000-E007F), zero-width (200B-200D,
    FEFF, 2060), bidi overrides (202A-202E, 2066-2069) -> surfaces ASCII-smuggling payloads
  * NFKC fold -> collapses many homoglyphs to their ASCII form
  * decode long base64/hex blobs and append the decoded text in-band -> surfaces encoded payloads
"""

import base64
import binascii
import re
import unicodedata

INVISIBLE = re.compile(
    "["
    "\U000E0000-\U000E007F"   # Unicode Tags block (ASCII smuggling)
    "\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD"  # Private Use Areas (used to interleave/hide text)
    "​-‍﻿⁠"  # zero-width space/joiner/non-joiner, BOM, word-joiner
    "‪-‮⁦-⁩"  # bidi embedding / override / isolate
    "]"
)
B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
HEX = re.compile(r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{2}){16,}(?![0-9a-fA-F])")


def _printable(b: bytes) -> str | None:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(c.isprintable() or c.isspace() for c in s)
    return s if s and printable / len(s) > 0.85 else None


def decode_blobs(text: str) -> list[str]:
    out = []
    for m in B64.finditer(text):
        blob = m.group()
        if len(blob) % 4:
            continue
        try:
            s = _printable(base64.b64decode(blob, validate=True))
        except (binascii.Error, ValueError):
            s = None
        if s:
            out.append(s)
    for m in HEX.finditer(text):
        try:
            s = _printable(bytes.fromhex(m.group()))
        except ValueError:
            s = None
        if s:
            out.append(s)
    return out


def normalize_input(text: str) -> str:
    had_invis = bool(INVISIBLE.search(text))
    t = INVISIBLE.sub("", text)
    t = unicodedata.normalize("NFKC", t)
    decoded = decode_blobs(t)
    extra = ""
    if had_invis:
        extra += "\n[normalizer: hidden characters were removed from this content]"
    for d in decoded:
        extra += f"\n[normalizer: decoded embedded blob]: {d}"
    return t + extra


if __name__ == "__main__":  # quick self-check
    import sys
    print(normalize_input(sys.stdin.read()))
