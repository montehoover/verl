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
    ]

    for email_str in emails_to_test:
        print(f"'{email_str}' is valid: {is_valid_email(email_str)}")
