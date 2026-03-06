import re

# Zero-width & common invisible chars
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\uFEFF]")

# Collapse spaces/tabs, keep newlines
SPACES_RE = re.compile(r"[ \t]+")

# Too many newlines -> max two
MANY_NEWLINES_RE = re.compile(r"\n{3,}")

# URLs anywhere in text (we will keep only path, drop query/domain)
URL_RE = re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)

# Common Arabic punctuation variants (safe, optional)
ARABIC_COMMA_RE = re.compile("،")
ARABIC_SEMICOLON_RE = re.compile("؛")


def normalize_lite(text: str) -> str:
    """
    Safe normalization for ticket text:
    - remove invisible chars
    - normalize whitespace
    - normalize Arabic punctuation (optional)
    - sanitize URLs (keep only path; drop query string)
    """
    if not text:
        return ""

    t = str(text)

    # Remove invisible chars that can break matching
    t = ZERO_WIDTH_RE.sub("", t)

    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize Arabic punctuation (safe)
    t = ARABIC_COMMA_RE.sub(",", t)
    t = ARABIC_SEMICOLON_RE.sub(";", t)

    # Whitespace cleanup
    t = SPACES_RE.sub(" ", t)
    t = MANY_NEWLINES_RE.sub("\n\n", t)

    return t.strip()