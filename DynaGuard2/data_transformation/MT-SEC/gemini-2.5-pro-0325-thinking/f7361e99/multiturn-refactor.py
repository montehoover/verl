import re

def extract_email_domain(email_address: str) -> str | None:
    """
    Extracts the domain from an email address using a regular expression.

    The function uses a regular expression to validate the email address structure
    and extract the domain part. The expected email format is broadly
    `local_part@domain_part`, where the `domain_part` must contain at least
    one dot ('.') and a top-level domain (TLD) of at least two alphabetic
    characters (e.g., 'example.com', 'sub.example.co.uk').

    Args:
        email_address: The email address string to be processed.

    Returns:
        The domain part of the email address as a string if the input
        matches the defined email pattern. Returns None if the input
        is not a valid email address according to this pattern.

    Examples:
        >>> extract_email_domain("user@example.com")
        'example.com'
        >>> extract_email_domain("firstname.lastname@sub.example.co.uk")
        'sub.example.co.uk'
        >>> extract_email_domain("user@example.c") # TLD too short
        None
        >>> extract_email_domain("user@localhost") # Lacks a TLD structure like '.com'
        None
        >>> extract_email_domain("invalid-email")
        None
        >>> extract_email_domain("user@.com") # Domain name part missing
        None
        >>> extract_email_domain("@example.com") # Local part missing
        None
    """
    # Guard clause: Validate that the input is a string.
    # Non-string inputs are considered invalid for email processing.
    if not isinstance(email_address, str):
        return None

    # Guard clause: Basic check for the presence of '@'.
    # An email address must contain an '@' symbol. This is a quick check
    # before applying the more complex regular expression.
    if "@" not in email_address:
        return None

    # Regular expression for validating an email address and capturing its domain.
    # The `re.fullmatch` function is used to ensure the entire string matches this pattern.
    #
    # Pattern Breakdown:
    #   `[a-zA-Z0-9._%+-]+` : Matches the local part (e.g., 'username') before the '@'.
    #                         It allows one or more occurrences of:
    #                         - `a-z`, `A-Z`: Alphabetic characters.
    #                         - `0-9`: Numeric digits.
    #                         - `._%+-`: Special characters period, underscore, percent, plus, hyphen.
    #   `@`                   : Matches the literal '@' symbol, separating local and domain parts.
    #   `(`                   : Begins a capturing group for the domain part. This allows
    #                         us to extract just the domain.
    #     `[a-zA-Z0-9.-]+`  : Matches the domain name labels (e.g., 'example', 'sub.example').
    #                         It allows one or more occurrences of alphanumeric characters,
    #                         dots (for subdomains), and hyphens.
    #     `\.`                : Matches the literal dot ('.') that separates the domain name
    #                         from the top-level domain (TLD).
    #     `[a-zA-Z]{2,}`    : Matches the top-level domain (TLD) (e.g., 'com', 'org', 'net').
    #                         It requires at least two alphabetic characters.
    #   `)`                   : Ends the capturing group for the domain.
    email_pattern = r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    
    # Attempt to match the entire email_address string against the defined pattern.
    match_object = re.fullmatch(email_pattern, email_address)

    # If the `email_address` does not conform to the `email_pattern`,
    # `match_object` will be None, indicating an invalid email format.
    if not match_object:
        return None

    # If a match is found, `match_object` is not None.
    # The domain is in the first capturing group (group 1).
    # group(0) would be the entire matched string (the full email address).
    return match_object.group(1)
