import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# UAE phone formats:
# 05XXXXXXXX (10 digits total)
UAE_LOCAL_MOBILE_RE = re.compile(r"\b05\d{8}\b")

# +9715XXXXXXXX or 9715XXXXXXXX (UAE mobile without leading 0)
UAE_INT_MOBILE_RE = re.compile(r"\b(?:\+971|971)\s?5\d{8}\b")

# Generic international phone numbers ONLY when prefixed by '+'
# (This avoids masking IDs like 202600487657)
INTL_PLUS_PHONE_RE = re.compile(r"\+\d{7,15}\b")

# Emirates ID (common format 784-XXXX-XXXXXXX-X, allow digits only with dashes)
EMIRATES_ID_RE = re.compile(r"\b784-\d{4}-\d{7}-\d\b")

# Mask eID query parameter in URLs (very relevant for you)
EID_PARAM_RE = re.compile(r"(?i)(\beid\s*=\s*)(\d{6,})")

# Full URL like https://host/path?x=1  OR http://host/path
FULL_URL_RE = re.compile(r"^[a-zA-Z]+://[^/]+(?P<path>/.*)$")

# Mask sensitive query params if you ever keep them (we will drop queries anyway)
EID_PARAM_RE = re.compile(r"(?i)(\beid\s*=\s*)(\S+)")


def sanitize_endpoint_line(line: str) -> str:
    """
    Sanitize a single line that may contain:
    - full URL
    - gateway path
    - plain endpoint path
    Goal: keep path, drop query string, preserve line content.
    """
    if not line:
        return ""

    s = line.strip()
    if not s:
        return ""

    # If it's a full URL, keep only path
    m = FULL_URL_RE.match(s)
    if m:
        s = m.group("path")

    # Drop query string (args=..., eID=..., etc.)
    s = s.split("?", 1)[0]

    # Final trim
    return s.strip()


def sanitize_endpoints_text(text: str) -> str:
    """
    Sanitize multi-line API/operation text safely:
    - keep line breaks
    - sanitize each line independently
    - preserve separators like '-----'
    """
    if not text:
        return ""

    lines = str(text).splitlines()
    out_lines = []
    for ln in lines:
        # Keep visual separators as-is
        if set(ln.strip()) <= {"-", "_"} and ln.strip():
            out_lines.append(ln.strip())
            continue

        cleaned = sanitize_endpoint_line(ln)
        # Keep empty lines to preserve structure (optional). If you prefer, skip them.
        out_lines.append(cleaned)

    # Do NOT remove '/getStepInfo' lines; keep them.
    return "\n".join(out_lines).strip()

def mask_pii(text: str) -> str:
    if not text:
        return ""

    t = str(text)

    # High-confidence replacements only
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = IP_RE.sub("[IP]", t)

    t = EMIRATES_ID_RE.sub("[EMIRATES_ID]", t)

    t = UAE_LOCAL_MOBILE_RE.sub("[PHONE]", t)
    t = UAE_INT_MOBILE_RE.sub("[PHONE]", t)
    t = INTL_PLUS_PHONE_RE.sub("[PHONE]", t)

    # Mask eID=123... anywhere in free text too
    t = EID_PARAM_RE.sub(r"\1[ID]", t)

    return t