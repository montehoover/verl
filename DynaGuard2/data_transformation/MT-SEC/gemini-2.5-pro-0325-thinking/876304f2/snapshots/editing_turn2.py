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

def get_email_username(email: str) -> str | None:
    """
    Extracts the username from a valid email address.

    Args:
        email: The email string.

    Returns:
        The username part of the email if valid, otherwise None.
    """
    if is_valid_email(email):
        # Regex to capture the username part (everything before '@')
        match = re.match(r'^([^@]+)@', email)
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
            username = get_email_username(email_str)
            print(f"  Username: {username}")
        else:
            # Attempt to get username even if invalid, to test get_email_username's None return
            username = get_email_username(email_str)
            print(f"  Username (attempt on invalid): {username}")
        print("-" * 20)
