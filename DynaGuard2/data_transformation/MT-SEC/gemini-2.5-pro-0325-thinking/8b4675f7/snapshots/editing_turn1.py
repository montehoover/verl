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

if __name__ == '__main__':
    # Test cases
    print(f"'test@example.com' is valid: {is_valid_email('test@example.com')}")  # Expected: True
    print(f"'invalid-email' is valid: {is_valid_email('invalid-email')}")    # Expected: False
    print(f"'test@example' is valid: {is_valid_email('test@example')}")        # Expected: False
    print(f"'@example.com' is valid: {is_valid_email('@example.com')}")        # Expected: False
    print(f"'test@.com' is valid: {is_valid_email('test@.com')}")            # Expected: False
    print(f"'test@example..com' is valid: {is_valid_email('test@example..com')}") # Expected: False
    print(f"'another.test-email+tag@sub.domain.co.uk' is valid: {is_valid_email('another.test-email+tag@sub.domain.co.uk')}") # Expected: True
    print(f"'' is valid: {is_valid_email('')}") # Expected: False
    print(f"None is valid: {is_valid_email(None)}") # Expected: False (due to try-except, though re.match would raise TypeError)
    print(f"123 is valid: {is_valid_email(123)}") # Expected: False (due to try-except, though re.match would raise TypeError)
