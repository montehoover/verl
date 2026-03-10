def is_valid_email(email: str) -> bool:
    """
    Return True if 'email' appears to be a basic valid email address format:
    - contains exactly one '@'
    - has at least one '.' in the domain part (after the '@')
    """
    if not isinstance(email, str):
        return False

    email = email.strip()

    # Must contain exactly one '@'
    if email.count("@") != 1:
        return False

    local, domain = email.split("@")

    # Local and domain parts must be non-empty
    if not local or not domain:
        return False

    # Domain must contain at least one '.'
    if "." not in domain:
        return False

    return True
