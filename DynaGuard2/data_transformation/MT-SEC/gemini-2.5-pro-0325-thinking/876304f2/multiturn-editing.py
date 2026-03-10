import re

def is_valid_email(email: str) -> bool:
    """
    Checks if the given string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    # Regular expression for validating an email
    # This is a common regex, but email validation can be complex.
    # For production systems, consider using a well-vetted library.
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(regex, email):
        return True
    else:
        return False

def get_email_domain(addr: str) -> str | None:
    """
    Extracts the domain from a valid email address.

    Args:
        addr: The email string.

    Returns:
        The domain part of the email if valid, otherwise None.
    """
    if is_valid_email(addr):
        # Regex to capture the domain part (everything after '@')
        match = re.match(r'^[a-zA-Z0-9._%+-]+@([^@]+)$', addr)
        if match:
            return match.group(1)
    return None

if __name__ == '__main__':
    # Example Usage
    emails_to_test = [
        "test@example.com",
        "valid.email+filter@example.co.uk",
        "invalid-email",
        "another@example",
        "user@.com",
        "@domain.com",
        "user@domain.",
        "user@domain.c",
        "username_only@", # Should be invalid by is_valid_email
        "test.user@sub.example.com"
    ]

    for email_str in emails_to_test:
        is_valid = is_valid_email(email_str)
        print(f"'{email_str}' is valid: {is_valid}")
        if is_valid:
            domain = get_email_domain(email_str)
            print(f"  Domain: {domain}")
        else:
            # Attempt to get domain even if invalid, to test get_email_domain's None return
            domain = get_email_domain(email_str)
            print(f"  Domain (attempt on invalid): {domain}")
        print("-" * 20)
