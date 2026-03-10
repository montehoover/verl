import re
from typing import Optional

def retrieve_domain_from_email(email_address: str) -> Optional[str]:
    """
    Extracts the domain from a valid email address.

    Args:
        email_address: The email string to validate and extract the domain from.

    Returns:
        The domain part of the email if valid, None otherwise.
    """
    # Regex to validate email and capture the domain part.
    # Domain part is everything after the '@' symbol.
    pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    match = re.match(pattern, email_address)
    if match:
        return match.group(1)  # Return the captured domain
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
        "another.test@sub.example.org",
    ]

    for email_str in emails_to_test:
        domain = retrieve_domain_from_email(email_str)
        if domain:
            print(f"Email: '{email_str}', Domain: '{domain}'")
        else:
            print(f"Email: '{email_str}' is invalid or no domain found.")
