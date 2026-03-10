import re

def is_valid_email(email: str) -> bool:
    """
    Checks if a given email address is valid.

    Args:
        email: The email address string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    # Regular expression for validating an email
    # This regex is a common one, but email validation can be complex.
    # It checks for a basic structure: something@something.something
    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(regex, email):
        return True
    else:
        return False

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
        print(f"'{email_str}' is valid: {is_valid_email(email_str)}")
