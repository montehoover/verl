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

def split_email(email: str) -> tuple[str, str]:
    """
    Splits a valid email address into its username and domain.

    Args:
        email: The email string to split. Assumes email is valid.

    Returns:
        A tuple containing the username and domain.
    """
    parts = email.split('@', 1)
    return parts[0], parts[1]

def get_email_domain(addr: str) -> str | None:
    """
    Extracts the domain part of an email address using regular expressions.

    Args:
        addr: The email string.

    Returns:
        The domain part of the email if valid, otherwise None.
    """
    # Regex to validate email and capture the domain part
    # This regex is similar to the one in is_valid_email but includes a capturing group for the domain.
    pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    match = re.match(pattern, addr)
    if match:
        return match.group(1)  # Return the captured domain
    return None

if __name__ == '__main__':
    # Example Usage for is_valid_email
    print("--- Email Validation ---")
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

    print("\n--- Email Splitting ---")
    valid_emails_for_splitting = [
        "test@example.com",
        "test.user@sub.example.co.uk",
        "another_test@example.org",
        "user@localhost",
    ]

    for email_str in valid_emails_for_splitting:
        if is_valid_email(email_str): # Ensure we only split valid emails for this demo
            username, domain = split_email(email_str)
            print(f"Email: '{email_str}' -> Username: '{username}', Domain: '{domain}'")
        else:
            # This case should ideally not be hit if we only pass valid emails
            print(f"'{email_str}' is not a valid email for splitting.")

    print("\n--- Get Email Domain ---")
    emails_for_domain_extraction = [
        "test@example.com",
        "invalid-email",
        "test.user@sub.example.co.uk",
        "another_test@example.org",
        "user@localhost",
        "user@192.168.1.1",
        "user@example..com",
        ".user@example.com",
        "user.@example.com",
        "user@.example.com",
        "user@example.c",
        "user@example.",
        "",
        "test@example.com ",
        " test@example.com",
    ]

    for email_str in emails_for_domain_extraction:
        domain = get_email_domain(email_str)
        if domain:
            print(f"Email: '{email_str}' -> Domain: '{domain}'")
        else:
            print(f"Email: '{email_str}' -> Invalid or no domain found")
