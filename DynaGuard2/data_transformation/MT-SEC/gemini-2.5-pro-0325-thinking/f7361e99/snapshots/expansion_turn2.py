import re

def is_valid_email_format(email: str) -> bool:
    """
    Checks if a given string is a valid email format.

    Args:
        email: The string to check.

    Returns:
        True if the email is in a valid format, False otherwise.
    """
    # Regular expression for basic email validation
    # This regex is a common one, but email validation can be complex.
    # For production systems, consider using a more robust library if available.
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True
    else:
        return False

def get_email_username(email: str) -> str | None:
    """
    Extracts the username from an email address.

    Args:
        email: The email address string.

    Returns:
        The username part of the email, or None if the email is invalid
        or does not contain an '@' symbol.
    """
    if not is_valid_email_format(email):
        return None
    try:
        username, _ = email.split('@', 1)
        return username
    except ValueError:
        # This case should ideally be caught by is_valid_email_format,
        # but as a safeguard if is_valid_email_format were to change
        # or if an email without '@' somehow passed.
        return None

if __name__ == '__main__':
    # Example Usage
    emails_to_test = [
        "test@example.com",
        "test.user@example.co.uk",
        "test_user123@example-domain.com",
        "invalid-email",
        "test@.com",
        "test@domain.",
        "@domain.com",
        "test@domain.c",
        "another.test@sub.example.org",
    ]

    for email_str in emails_to_test:
        is_valid = is_valid_email_format(email_str)
        print(f"'{email_str}' is valid: {is_valid}")
        username = get_email_username(email_str)
        if username:
            print(f"  Username: {username}")
        else:
            print(f"  Could not extract username from '{email_str}'")
