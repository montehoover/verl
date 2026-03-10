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

def get_email_domain(mail_address: str) -> str | None:
    """
    Extracts the domain from a valid email address.

    Args:
        mail_address: The email string.

    Returns:
        The domain part of the email if valid, otherwise None.
    """
    if not is_valid_email(mail_address):
        return None
    
    # Regex to capture the domain part (everything after '@')
    # This assumes the email is already validated by is_valid_email
    match = re.match(r"^[^@]+@(.+)$", mail_address)
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

    # Test cases for get_email_domain
    print("\n--- get_email_domain tests ---")
    print(f"Domain for 'test@example.com': {get_email_domain('test@example.com')}") # Expected: example.com
    print(f"Domain for 'user.name+tag@example.co.uk': {get_email_domain('user.name+tag@example.co.uk')}") # Expected: example.co.uk
    print(f"Domain for 'another.test-email+tag@sub.domain.co.uk': {get_email_domain('another.test-email+tag@sub.domain.co.uk')}") # Expected: sub.domain.co.uk
    print(f"Domain for 'invalid-email': {get_email_domain('invalid-email')}") # Expected: None
    print(f"Domain for '@example.com': {get_email_domain('@example.com')}") # Expected: None
    print(f"Domain for 'test@': {get_email_domain('test@')}") # Expected: None (is_valid_email will fail)
    print(f"Domain for '': {get_email_domain('')}") # Expected: None
    print(f"Domain for None: {get_email_domain(None)}") # Expected: None
