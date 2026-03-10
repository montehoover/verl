"""
Utilities to validate and extract the domain portion of an email address.

This module exposes a single public function:
- retrieve_email_domain: Returns the domain from a valid email address or None.

Implementation details:
- Splitting, validation, and extraction are done by pure helper functions
  to improve readability, testability, and maintainability.
"""

import re
from typing import Optional, Tuple

# Domain regex:
# - One or more labels separated by dots
#   - Each label starts and ends with alphanumeric, may contain hyphens in
#     between
#   - Final TLD is alphabetic and at least 2 characters
_DOMAIN_RE = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+(?:[A-Za-z]{2,})$"
)

# Simplified local-part regex (commonly permitted characters)
_LOCAL_PART_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+$")


def _split_email(email_str: str) -> Optional[Tuple[str, str]]:
    """
    Split an email-like string into local and domain parts.

    This is a pure function:
    - It does not mutate any external state.
    - It raises no exceptions.

    Args:
        email_str (str): The candidate email string.

    Returns:
        Optional[Tuple[str, str]]: (local_part, domain) if the input contains a
        single '@' separating non-empty parts; otherwise, None.
    """
    if not isinstance(email_str, str):
        return None

    stripped_email = email_str.strip()
    if stripped_email.count("@") != 1:
        return None

    local_part, domain = stripped_email.split("@", 1)
    if not local_part or not domain:
        return None

    return local_part, domain


def _is_valid_domain(domain: str) -> bool:
    """
    Validate a domain string against a conservative DNS-like pattern.

    Rules enforced:
    - Composed of labels separated by dots.
    - Each label starts and ends with alphanumeric characters.
    - Labels can include hyphens in the middle.
    - Final TLD is alphabetic and at least two characters long.

    Args:
        domain (str): The domain candidate.

    Returns:
        bool: True if valid; otherwise, False.
    """
    if not isinstance(domain, str) or not domain:
        return False

    return _DOMAIN_RE.fullmatch(domain) is not None


def _is_valid_local_part(local_part: str) -> bool:
    """
    Validate an email local-part using a simplified permissive pattern.

    Args:
        local_part (str): The local-part candidate (before '@').

    Returns:
        bool: True if valid; otherwise, False.
    """
    if not isinstance(local_part, str) or not local_part:
        return False

    return _LOCAL_PART_RE.fullmatch(local_part) is not None


def retrieve_email_domain(email_str: str) -> Optional[str]:
    """
    Extract the domain portion from a well-formed email address.

    Orchestrates pure helper functions for splitting and validation. Returns
    None if the input is invalid. This function does not raise exceptions.

    Args:
        email_str (str): The email address to parse.

    Returns:
        Optional[str]: The domain if the email is valid; otherwise, None.
    """
    try:
        split_parts = _split_email(email_str)
        if not split_parts:
            return None

        local_part, domain = split_parts

        if not _is_valid_local_part(local_part):
            return None

        if not _is_valid_domain(domain):
            return None

        return domain
    except Exception:
        # Never propagate exceptions as per requirements.
        return None
