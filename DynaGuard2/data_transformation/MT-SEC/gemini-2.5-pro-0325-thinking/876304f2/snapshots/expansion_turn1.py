import re

def is_valid_email(email: str) -> bool:
    """
    Checks if a given string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    # A basic regex for email validation.
    # This regex is not exhaustive but covers most common cases.
    # For a more robust solution, a dedicated library might be better.
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True
    return False

if __name__ == '__main__':
    # Example Usage
    emails_to_test = [
        "test@example.com",
        "invalid-email",
        "test.user@sub.example.co.uk",
        "another_test@example.org",
        "user@localhost", # often considered valid
        "user@192.168.1.1", # valid if IP addresses are allowed
        "user@example..com", # invalid
        ".user@example.com", # invalid
        "user.@example.com", # invalid
        "user@.example.com", # invalid
        "user@example.c", # invalid (TLD too short)
        "user@example.", # invalid
        "", # empty string
        "test@example.com ", # trailing space
        " test@example.com", # leading space
    ]

    for email_str in emails_to_test:
        is_valid = is_valid_email(email_str)
        print(f"'{email_str}' is valid: {is_valid}")
