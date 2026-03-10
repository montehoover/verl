import re
from typing import Optional

def extract_username(email: str) -> Optional[str]:
    """
    Validates an email and extracts the username part if valid.

    Args:
        email: The email string to validate and extract from.

    Returns:
        The username part of the email if valid, None otherwise.
    """
    # Regex to validate email and capture the username part.
    # Username part is everything before the '@' symbol.
    pattern = r"^([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    match = re.match(pattern, email)
    if match:
        return match.group(1)  # Return the captured username
    else:
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
    ]

    for email_str in emails_to_test:
        username = extract_username(email_str)
        if username:
            print(f"Email: '{email_str}', Username: '{username}'")
        else:
            print(f"Email: '{email_str}' is invalid.")
