"""Utilities for validating email addresses and extracting domains.

This module exposes a single helper function, `extract_email_domain`,
which validates an email address with a regular expression and returns
its domain part (the substring after the '@' character).
"""
import re
from typing import Optional

# Pre-compiled regular expression for validating an email address and capturing
# the domain part. We intentionally use a pragmatic subset of RFC 5322 for the
# local-part (dot-atom) and support traditional DNS names or bracketed IP
# literals for the domain.
#
# Pattern overview (using re.VERBOSE for readability):
# - ^ and $ anchor the pattern to match the entire input string.
# - (?P<local>...) captures the local-part (before '@') using a "dot-atom"
#   syntax composed of allowed characters separated by dots.
# - @ literal separates local and domain parts.
# - (?P<domain>...) captures the domain part (after '@'). It supports:
#   • A traditional DNS name:
#       - Labels start/end with an alphanumeric character.
#       - Hyphens allowed inside labels (not at the edges).
#       - At least one dot to ensure a TLD-like structure (e.g., example.com).
#   • A domain-literal in square brackets:
#       - IPv6: [IPv6:...]
#       - IPv4: [x.x.x.x] with each octet in 0–255.
_EMAIL_REGEX = re.compile(
    r"""
    ^

    (?P<local>
        [A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+          # atom
        (?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*   # dot-atom (zero or more)
    )

    @

    (?P<domain>
        (?:                                      # DNS name:
            [A-Za-z0-9]                          # label start (alnum)
            (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?   # label body (no edge hyphen)
            (?:\.
                [A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?
            )+                                   # one or more dots (needs TLD)
        )
        |
        (?:                                      # Domain-literal:
            \[
                (?:                              # IPv6 or IPv4 inside brackets
                    IPv6:[A-Fa-f0-9:.]+
                    |
                    (?:25[0-5]|2[0-4]\d|1?\d?\d) # IPv4 first octet
                    (?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}
                )
            \]
        )
    )

    $
    """,
    re.VERBOSE,
)


def extract_email_domain(email: str) -> Optional[str]:
    """Return the domain part of an email address if the address is valid.

    The function validates the input using a compiled regular expression and
    returns the substring after the '@' symbol (the domain) when the input
    resembles a conventional email address. If the input is not a string, is
    empty after trimming, or does not match the expected pattern, the function
    returns None.

    This function is designed to be safe to call: it does not raise exceptions.

    Args:
        email: The input email address to be processed.

    Returns:
        The domain part (everything after '@') when the email is valid.
        Otherwise, None.

    Examples:
        >>> extract_email_domain("user@example.com")
        'example.com'
        >>> extract_email_domain("USER+tag@sub.example.co.uk")
        'sub.example.co.uk'
        >>> extract_email_domain("not-an-email")
        >>> extract_email_domain("  user@example.com  ")
        'example.com'
    """
    # Guard clause: ensure the input is a string.
    if not isinstance(email, str):
        return None

    # Guard clause: trim whitespace and verify non-empty content.
    normalized_email = email.strip()
    if not normalized_email:
        return None

    # Guard clause: a quick check to avoid regex if '@' is missing.
    if "@" not in normalized_email:
        return None

    # Validate the entire email and capture the domain part if valid.
    match = _EMAIL_REGEX.fullmatch(normalized_email)
    if not match:
        return None

    # Return the captured 'domain' part of the email address.
    return match.group("domain")
