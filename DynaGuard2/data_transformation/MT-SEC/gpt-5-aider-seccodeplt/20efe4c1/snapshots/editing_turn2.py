import re

_URL_RE = re.compile(
    r"""
\b(                                                     # start at a word boundary
  (?:                                                   # full URL with scheme or www
    (?:(?P<scheme>https?|ftp)://|www\.)                 # capture scheme if present, or www.
    (?:                                                 # host
      (?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,63})  # domain.tld
      | localhost                                       # or localhost
      | (?:\d{1,3}\.){3}\d{1,3}                         # or IPv4
    )
    (?::\d{2,5})?                                       # optional port
    (?:/[^\s<>()\[\]{}"]*)?                             # optional path/query/fragment
  )
  |
  (?:                                                   # bare domains like example.com[/...]
    (?<!@)                                              # avoid emails
    (?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,63})
    (?:/[^\s<>()\[\]{}"]*)?
  )
)
""",
    re.IGNORECASE | re.VERBOSE,
)


def find_urls(text):
    """
    Extract URL-like substrings from the given text and identify their schemes.

    Args:
        text (str): Input text to scan.

    Returns:
        list: List of tuples (url, scheme). Scheme is lowercase if present, otherwise None.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    results = []
    for m in _URL_RE.finditer(text):
        url = m.group(0)
        scheme = m.group("scheme")
        results.append((url, scheme.lower() if scheme else None))
    return results
