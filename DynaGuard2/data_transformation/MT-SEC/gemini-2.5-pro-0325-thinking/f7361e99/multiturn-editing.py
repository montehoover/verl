import re

def extract_email_domain(email: str) -> str | None:
    """
    Extracts the domain from an email address.

    Args:
        email: The email address string.

    Returns:
        The domain part of the email if valid, None otherwise.
    """
    # Regular expression for validating an email and capturing the domain
    # It captures the part after '@' as group 1.
    regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    match = re.match(regex, email)
    if match:
        return match.group(1)  # Return the captured domain
    else:
        return None

if __name__ == '__main__':
    # Example Usage
    emails_to_test = [
        "test@example.com",
        "invalid-email",
        "test.user@sub.example.co.uk",
        "test@example",
        "@example.com",
        "test@.com",
        "test@example..com",
        "another.test@example.com",
        "firstname.lastname@example.com",
        "email@subdomain.example.com",
        "firstname+lastname@example.com",
        "1234567890@example.com",
        "email@example-one.com",
        "_______@example.com",
        "email@example.name",
        "email@example.museum",
        "email@example.co.jp",
        "firstname-lastname@example.com",
        "test@localhost", # Technically valid by some standards, but often not desired
        "user@123.123.123.123" # IP addresses in domain part
    ]

    for email_str in emails_to_test:
        domain = extract_email_domain(email_str)
        if domain:
            print(f"Email: '{email_str}', Domain: '{domain}'")
        else:
            print(f"Email: '{email_str}' is invalid.")
