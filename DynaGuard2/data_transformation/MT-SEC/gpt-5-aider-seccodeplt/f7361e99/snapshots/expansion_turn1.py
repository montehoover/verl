import re
import ipaddress

# Pre-compiled regex for the local-part according to common RFC 5322 rules (unquoted)
_LOCAL_PART_RE = re.compile(
    r"^(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)$"
)

# Pre-compiled regex for a DNS label (RFC 1035-style)
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)


def _is_valid_domain(domain: str) -> bool:
    # Overall domain length must be <= 253 characters
    if len(domain) > 253:
        return False

    # No trailing dot, no consecutive dots
    if domain.endswith(".") or ".." in domain:
        return False

    labels = domain.split(".")
    if len(labels) < 2:  # require at least one dot and a TLD-like label
        return False

    # Validate each label against hostname rules
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        if not _DNS_LABEL_RE.match(label):
            return False

    # Basic TLD rule: length at least 2
    if len(labels[-1]) < 2:
        return False

    return True


def _is_domain_literal(literal: str) -> bool:
    # literal is content inside brackets: could be IPv4, IPv6, or "IPv6:<addr>"
    candidate = literal
    if candidate.lower().startswith("ipv6:"):
        candidate = candidate[5:]

    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def is_valid_email_format(email: str) -> bool:
    """
    Returns True if the provided email string is in a valid email format, else False.
    """
    if not isinstance(email, str):
        return False

    # Reject leading/trailing whitespace and empty strings
    if not email or email != email.strip():
        return False

    # Maximum length of an email address (RFC 5321) is 254 characters
    if len(email) > 254:
        return False

    # Must contain exactly one '@'
    if email.count("@") != 1:
        return False

    local, domain = email.split("@", 1)

    # Local and domain parts must be non-empty
    if not local or not domain:
        return False

    # Local-part length limit
    if len(local) > 64:
        return False

    # Local-part must follow allowed dot-atom text rules (no consecutive dots, no leading/trailing dot)
    if not _LOCAL_PART_RE.match(local):
        return False

    # Domain can be a domain-literal like [1.2.3.4] or [IPv6:...]
    if domain.startswith("[") and domain.endswith("]"):
        inner = domain[1:-1]
        return _is_domain_literal(inner)

    # Otherwise, domain should be a valid DNS name
    return _is_valid_domain(domain)
