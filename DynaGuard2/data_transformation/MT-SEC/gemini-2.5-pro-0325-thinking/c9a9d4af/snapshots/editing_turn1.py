import re

def validate_email(email: str) -> bool:
    """
    Checks whether a given email is formatted correctly.

    Args:
        email: The email string to validate.

    Returns:
        True if the email has a valid format, False otherwise.
    """
    # A common regex for email validation. It's not perfect but covers most cases.
    # For a more robust solution, a library like 'email_validator' might be preferred.
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
        "invalid-email",
        "test@.com",
        "test@domain.",
        "@domain.com",
        "test@domain.c",
    ]

    for email_str in emails_to_test:
        is_valid = validate_email(email_str)
        print(f"'{email_str}' is valid: {is_valid}")
