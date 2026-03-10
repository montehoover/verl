import re

def fetch_email_domain(mail_id: str) -> str | None:
    """
    Extracts the domain portion from an email address.

    Args:
        mail_id: The email address to parse.

    Returns:
        The domain part of the email (e.g., "example.com") or None if
        the email address is not well-formed or no domain is found.
    """
    # Regex to capture the domain part of an email address.
    # It looks for one or more characters (username part) followed by "@",
    # then captures the domain part (e.g., "example.com").
    # The domain part must contain at least one dot and end with at least two letters.
    match = re.fullmatch(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", mail_id)
    if match:
        # The first captured group (index 1) is the domain.
        return match.group(1)
    return None

if __name__ == '__main__':
    # Example Usage
    emails_to_test = [
        "user@example.com",
        "test.user@sub.example.co.uk",
        "another_user123@domain.info",
        "invalid-email",  # Should return None
        "user@.com",  # Should return None
        "user@domain.",  # Should return None
        "@example.com",  # Should return None
        "user@localhost", # Should return None (no TLD)
        "user@example.c", # Should return None (TLD too short)
        "first.last@department.big-company.com",
    ]

    for email in emails_to_test:
        domain = fetch_email_domain(email)
        print(f"Email: \"{email}\" -> Domain: \"{domain}\"")

    # Expected output:
    # Email: "user@example.com" -> Domain: "example.com"
    # Email: "test.user@sub.example.co.uk" -> Domain: "sub.example.co.uk"
    # Email: "another_user123@domain.info" -> Domain: "domain.info"
    # Email: "invalid-email" -> Domain: "None"
    # Email: "user@.com" -> Domain: "None"
    # Email: "user@domain." -> Domain: "None"
    # Email: "@example.com" -> Domain: "None"
    # Email: "user@localhost" -> Domain: "None"
    # Email: "user@example.c" -> Domain: "None"
    # Email: "first.last@department.big-company.com" -> Domain: "department.big-company.com"
