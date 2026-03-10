import re
from typing import Dict, List


def extract_components(text: str) -> Dict[str, List[str]]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Words: sequences of letters (including Unicode letters), excluding digits and underscores
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)

    # Numbers: integers or decimals with optional leading sign
    numbers = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", text)

    # Special characters: any non-word, non-whitespace character (punctuation, symbols, etc.)
    special_chars = re.findall(r"[^\w\s]", text, flags=re.UNICODE)

    return {
        "words": words,
        "numbers": numbers,
        "special_chars": special_chars,
    }


def get_url_scheme(url: str) -> str:
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9+.-]*):", url)
    if match:
        return match.group(1).lower()
    return "no_scheme"


def analyze_scheme(url: str) -> bool:
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    # Find any scheme-like tokens in the string (e.g., "javascript:", "http:", "data:", etc.)
    for match in re.finditer(r"(?i)\b([a-z][a-z0-9+.-]*)\s*:", url):
        scheme = match.group(1).lower()

        # Explicitly harmful schemes
        if scheme in ("javascript", "vbscript"):
            return True

        # Data URLs are considered harmful unless explicitly whitelisted as image types
        if scheme == "data":
            # Slice from the start of the matched scheme to validate the data URL
            tail = url[match.start():]
            # Allow common raster image media types only
            if re.match(
                r"(?is)^data\s*:\s*image/(?:png|jpe?g|gif|bmp|webp)(?:;charset=[^;,]+)?(?:;base64)?\s*,",
                tail,
            ):
                # Allowed image data URL (not harmful)
                continue
            # Any other data: URL is potentially harmful
            return True

    # No harmful scheme detected
    return False
