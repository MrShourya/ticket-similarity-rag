import re


def normalize_text(value) -> str:
    """
    Cleans text fields from Excel.
    """

    if value is None:
        return ""

    text = str(value)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if text.strip().upper() in {"NA", "N/A", "NONE", "NULL"}:
        return ""

    return text.strip()