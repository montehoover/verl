import re

# Precompiled regex for performance and clarity.
# - Local part: dot-atom format with permitted characters.
# - Domain: one or more labels ending with a TLD (2-63 letters), or 'localhost'.
_DOT_ATOM = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_TLD = r"[A-Za-z]{2,63}"
_EMAIL_REGEX = re.compile(
    rf"^(?P<local>{_DOT_ATOM})@(?P<domain>(?:{_LABEL}\.)+{_TLD}|localhost)$",
    re.ASCII,
)

def retrieve_email_domain(email_str: str):
    """
    Extract the domain portion from a well-formed email address.

    Args:
        email_str (str): The email address to parse.

    Returns:
        str | None: The domain portion if the email is valid; otherwise None.
    """
    try:
        if not isinstance(email_str, str):
            return None

        s = email_str.strip()
        if not s:
            return None

        m = _EMAIL_REGEX.fullmatch(s)
        if not m:
            return None

        return m.group("domain")
    except Exception:
        # Must not raise; return None on any unexpected issue.
        return None
