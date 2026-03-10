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

def extract_components(text: str):
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]
