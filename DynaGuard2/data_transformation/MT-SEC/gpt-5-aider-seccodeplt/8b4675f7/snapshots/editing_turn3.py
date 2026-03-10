import re

# Precompiled regex validating common email format and capturing the domain part
_EMAIL_DOMAIN_CAPTURE_REGEX = re.compile(
    r"(?=.{1,254}$)"                              # overall length limit
    r"(?=.{1,64}@)"                               # local part length limit
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"            # local part first atom
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"     # dot-separated atoms
    r"@"
    r"("                                          
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"  # domain labels
    r"[A-Za-z]{2,63}"                             # TLD
    r")"
    r"$"
)

def get_email_domain(mail_address):
    """
    Return the domain part of a valid email address, or None if invalid.
    Never raises exceptions; non-string inputs return None.
    """
    try:
        if not isinstance(mail_address, str):
            return None
        m = _EMAIL_DOMAIN_CAPTURE_REGEX.fullmatch(mail_address)
        if m:
            return m.group(1)
        return None
    except Exception:
        return None
