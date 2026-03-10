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
    # Regular expression for validating an email address and capturing its domain.
    # - `[a-zA-Z0-9._%+-]+`: Matches the local part (before '@'). Allows alphanumeric
    #   characters and specific symbols (._%+-) one or more times.
    # - `@`: Matches the literal '@' symbol.
    # - `([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})`: This is the capturing group for the domain.
    #   - `[a-zA-Z0-9.-]+`: Matches the domain name labels (e.g., 'example', 'sub.example').
    #     Allows alphanumeric characters, dots, and hyphens.
    #   - `\.`: Matches the literal dot separating domain labels from the TLD.
    #   - `[a-zA-Z]{2,}`: Matches the top-level domain (TLD), requiring at least
    #     two alphabetic characters (e.g., 'com', 'org', 'uk').
    # `re.fullmatch` ensures that the entire `email_address` string matches this pattern.
    email_pattern = r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    match_object = re.fullmatch(email_pattern, email_address)

    # If the email address matches the pattern, extract and return the domain.
    if match_object:
        # group(1) corresponds to the first capturing group in the regex,
        # which is the domain part: ([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})
        return match_object.group(1)
    else:
        # If the email address does not conform to the pattern, return None.
        return None
