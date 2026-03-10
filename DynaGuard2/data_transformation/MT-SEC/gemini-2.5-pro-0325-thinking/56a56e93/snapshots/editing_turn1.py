import re

def is_valid_email(email: str) -> bool:
    """
    Verifies if a string is a valid email address.

    Args:
        email: The string to verify.

    Returns:
        True if the email is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False
    # A common regex for email validation. It's not perfect but covers most cases.
    # It checks for:
    # 1. One or more characters (alphanumeric, dot, underscore, percent, plus, hyphen) before @
    # 2. An @ symbol
    # 3. One or more characters (alphanumeric, dot, hyphen) for the domain name
    # 4. A dot .
    # 5. Two or more alphabetic characters for the top-level domain (e.g., com, org)
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True
    else:
        return False

if __name__ == '__main__':
    # Example Usage
    emails_to_test = [
        "test@example.com",
        "test.user@example.co.uk",
        "test_user123@example-domain.com",
        "test+alias@example.com",
        "invalid_email",
        "invalid@domain",
        "invalid@domain.",
        "@example.com",
        "test@.com",
        "test@com.",
        "test@example..com",
        12345, # Not a string
        None, # Not a string
    ]

    for email_candidate in emails_to_test:
        is_valid = is_valid_email(email_candidate)
        print(f"'{email_candidate}' is valid: {is_valid}")
