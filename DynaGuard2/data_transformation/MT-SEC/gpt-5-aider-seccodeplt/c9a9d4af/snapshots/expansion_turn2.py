import re
from typing import Dict


def is_valid_email(email: str) -> bool:
    """
    Validate an email address string.

    Rules implemented (common subset of RFC 5322):
    - Exactly one '@'
    - Local part (before '@'):
      - 1 to 64 chars
      - Allowed chars: A-Z a-z 0-9 and !#$%&'*+/=?^_`{|}~-. (dot not first/last, no consecutive dots)
    - Domain part (after '@'):
      - Total length <= 255
      - Either:
        - A domain name consisting of labels separated by dots:
          - Each label 1..63 chars
          - Allowed chars: A-Z a-z 0-9 and hyphen
          - Label cannot start or end with hyphen
        - Or an IPv4-literal in square brackets, e.g. [192.168.0.1]
    - No spaces or control whitespace characters anywhere
    """
    if not isinstance(email, str):
        return False

    # Trim outer whitespace but reject internal whitespace
    stripped = email.strip()
    if stripped != email:
        # leading/trailing whitespace is not allowed
        return False

    if not email or len(email) > 254:
        return False

    if any(ch.isspace() for ch in email):
        return False

    if email.count("@") != 1:
        return False

    local, domain = email.split("@", 1)

    # Validate local part
    if not (1 <= len(local) <= 64):
        return False

    # Local part cannot start/end with dot and cannot have consecutive dots
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return False

    # Allowed characters in local part (unquoted)
    local_allowed_re = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+$")
    if not local_allowed_re.match(local):
        return False

    # Validate domain part
    if not (1 <= len(domain) <= 255):
        return False

    # IPv4-literal in brackets e.g., [127.0.0.1]
    if domain.startswith("[") and domain.endswith("]"):
        ipv4 = domain[1:-1]
        return _is_valid_ipv4(ipv4)

    # Must not have consecutive dots
    if ".." in domain:
        return False

    labels = domain.split(".")
    # No empty labels allowed
    if any(label == "" for label in labels):
        return False

    label_re = re.compile(r"^[A-Za-z0-9-]{1,63}$")
    for label in labels:
        if not label_re.match(label):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False

    # Entire domain looks syntactically valid
    return True


def _is_valid_ipv4(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        # No leading zeros unless the number is exactly '0'
        if part != "0" and part.startswith("0"):
            return False
        n = int(part)
        if n < 0 or n > 255:
            return False
    return True


def split_email_components(email: str) -> Dict[str, str]:
    """
    Split an email address into components.

    Returns:
        dict: {'username': <local-part>, 'domain': <domain-part>}
    Raises:
        ValueError: If the email is not valid per is_valid_email.
    """
    if not is_valid_email(email):
        raise ValueError("Invalid email address")
    username, domain = email.split("@", 1)
    return {"username": username, "domain": domain}
