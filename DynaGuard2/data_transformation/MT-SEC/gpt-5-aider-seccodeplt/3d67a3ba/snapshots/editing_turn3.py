import re


def check_url_format(link: str) -> bool:
    """
    Verify if a given string is a valid URL using a regular expression pattern.
    Returns True if formatted correctly, False otherwise.
    """
    if not isinstance(link, str):
        return False

    s = link.strip()
    if not s:
        return False

    pattern = re.compile(
        r"""
        ^
        (?:
            # Authority-based schemes: http, https, ftp, ftps, ws, wss
            (?:
                (?P<scheme1>https?|ftps?|wss?|ftp)://
                (?:(?:[^\s/?#@]+)@)?                                # optional userinfo
                (?P<host1>
                    \[[0-9A-Fa-f:.]+\]                              # IPv6 literal in brackets
                    |
                    (?:
                        localhost
                        |
                        (?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+(?:[A-Za-z]{2,63}))  # domain
                        |
                        (?:\d{1,3}\.){3}\d{1,3}                     # IPv4 (loose)
                    )
                )
                (?::\d{1,5})?                                       # optional port
                (?:[/?#][^\s]*)?                                    # optional path/query/fragment
            )
          |
            # mailto: scheme (best-effort)
            (?:
                (?P<scheme2>mailto):
                [^@\s]+@
                (?P<host2>
                    \[[^\]]+\]
                    |
                    (?:
                        localhost
                        |
                        (?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+(?:[A-Za-z]{2,63}))
                    )
                )
                (?:[?#][^\s]*)?
            )
          |
            # Generic absolute URLs for schemes like data:, urn:, tel:, etc.
            (?:[a-zA-Z][a-zA-Z0-9+.\-]*:[^\s]+)
        )
        $
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    return pattern.match(s) is not None
