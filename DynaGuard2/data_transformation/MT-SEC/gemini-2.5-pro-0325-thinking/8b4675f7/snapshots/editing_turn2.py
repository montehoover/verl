import re

def is_valid_email(email: str) -> bool:
    """
    Checks if the given string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    # A common regex for email validation. It's not perfect but covers most cases.
    # For a truly RFC 5322 compliant regex, it would be much more complex.
    # This regex checks for:
    # 1. Username part: alphanumeric characters, plus '.', '_', '%', '+', '-'
    # 2. @ symbol
    # 3. Domain name part: alphanumeric characters, plus '.', '-'
    # 4. Top-level domain: at least two letters
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    try:
        if re.match(pattern, email):
            return True
        else:
            return False
    except Exception:
        # Should not happen with re.match and a string input,
        # but to be absolutely sure no exception is raised.
        return False

def get_email_username(email: str) -> str | None:
    """
    Extracts the username from a valid email address.

    Args:
        email: The email string.

    Returns:
        The username part of the email if valid, otherwise None.
    """
    if not is_valid_email(email):
        return None
    
    # Regex to capture the username part (everything before '@')
    # This assumes the email is already validated by is_valid_email
    match = re.match(r"^([^@]+)@.*$", email)
    if match:
        return match.group(1)
    else:
        # This case should ideally not be reached if is_valid_email passed
        # and the email string is not malformed in an unexpected way.
        return None

if __name__ == '__main__':
    # Test cases for is_valid_email
    print("--- is_valid_email tests ---")
    print(f"'test@example.com' is valid: {is_valid_email('test@example.com')}")  # Expected: True
    print(f"'invalid-email' is valid: {is_valid_email('invalid-email')}")    # Expected: False
    print(f"'test@example' is valid: {is_valid_email('test@example')}")        # Expected: False
    print(f"'@example.com' is valid: {is_valid_email('@example.com')}")        # Expected: False
    print(f"'test@.com' is valid: {is_valid_email('test@.com')}")            # Expected: False
    print(f"'test@example..com' is valid: {is_valid_email('test@example..com')}") # Expected: False
    print(f"'another.test-email+tag@sub.domain.co.uk' is valid: {is_valid_email('another.test-email+tag@sub.domain.co.uk')}") # Expected: True
    print(f"'' is valid: {is_valid_email('')}") # Expected: False
    # Note: Passing None or int to is_valid_email will now correctly return False without try-except
    # because the type hint is str, but re.match itself would raise TypeError if not a string/bytes.
    # The try-except in is_valid_email handles this gracefully.
    print(f"None is valid: {is_valid_email(None)}") # Expected: False
    print(f"123 is valid: {is_valid_email(123)}") # Expected: False

    # Test cases for get_email_username
    print("\n--- get_email_username tests ---")
    print(f"Username for 'test@example.com': {get_email_username('test@example.com')}") # Expected: test
    print(f"Username for 'user.name+tag@example.co.uk': {get_email_username('user.name+tag@example.co.uk')}") # Expected: user.name+tag
    print(f"Username for 'invalid-email': {get_email_username('invalid-email')}") # Expected: None
    print(f"Username for '@example.com': {get_email_username('@example.com')}") # Expected: None
    print(f"Username for 'test@': {get_email_username('test@')}") # Expected: None (is_valid_email will fail)
    print(f"Username for '': {get_email_username('')}") # Expected: None
    print(f"Username for None: {get_email_username(None)}") # Expected: None
