import re

_TOKEN_RE = re.compile(
    r"""
    (
        [^\W\d_]+(?:[-'][^\W\d_]+)*      # words (Unicode letters, optional hyphen/apostrophe)
      | [+-]?(?:\d+(?:\.\d+)?|\.\d+)     # numbers (integers or decimals, optional sign)
      | [^\w\s]|_                        # symbols (non-word, non-space, or underscore)
    )
    """,
    re.UNICODE | re.VERBOSE,
)

_URL_SCHEME_RE = re.compile(
    r'^\s*(?![A-Za-z]:[\\/])([A-Za-z][A-Za-z0-9+.-]*):',
    re.UNICODE | re.IGNORECASE,
)

def extract_components(text: str):
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]

def identify_url_scheme(url: str):
    m = _URL_SCHEME_RE.match(url)
    if m:
        return m.group(1).lower()
    return 'no_scheme'
