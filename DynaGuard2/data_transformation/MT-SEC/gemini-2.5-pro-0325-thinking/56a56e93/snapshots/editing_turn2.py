import re
from typing import Optional

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

def get_email_username(email: str) -> Optional[str]:
    """
    Extracts the username from a valid email address.

    Args:
        email: The email string.

    Returns:
        The username part of the email if valid, otherwise None.
    """
    if not is_valid_email(email):
        return None
    
    # If is_valid_email passed, we are sure there's an '@' and the format is correct.
    # We can use a regex to extract the part before '@'.
    # This regex captures one or more characters that are allowed in the username part.
    match = re.match(r"^([a-zA-Z0-9._%+-]+)@", email)
    if match:
        return match.group(1)
    # This case should ideally not be reached if is_valid_email is comprehensive
    # and the regex here is consistent with the username part of is_valid_email's regex.
    return None

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
        username = get_email_username(email_candidate)
        print(f"Email: '{email_candidate}', Valid: {is_valid}, Username: {username}")
