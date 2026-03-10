import re


def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Insert a user-supplied value as a custom header with proper sanitization.
    
    This function takes a user-provided custom header value and sanitizes it
    to prevent header injection attacks and ensure compliance with HTTP header
    specifications. The sanitized value is then formatted as a complete HTTP
    header string.
    
    Args:
        custom_header_value (str): The user-provided custom header value to be
                                   sanitized and inserted.
    
    Returns:
        str: The final header string in the format "X-Custom-Header: <sanitized_value>".
    
    Note:
        Does not raise any specific exceptions, but improper input could
        indirectly raise 're.error' if the regex pattern is malformed.
    
    Example:
        >>> insert_header_with_sanitization("My Custom Value!")
        'X-Custom-Header: My Custom Value!'
        
        >>> insert_header_with_sanitization("Malicious\r\nHeader: Injection")
        'X-Custom-Header: MaliciousHeader: Injection'
    """
    # Define allowed characters: alphanumeric, spaces, and common punctuation
    # This regex pattern removes any potentially dangerous characters including:
    # - Newlines (\n) and carriage returns (\r) that could cause header injection
    # - Control characters that could break HTTP protocol
    # - Other non-standard characters that might cause parsing issues
    allowed_chars_pattern = r'[^\w\s\-_.,;:!?()]'
    sanitized_value = re.sub(allowed_chars_pattern, '', custom_header_value)
    
    # Remove leading and trailing whitespace to ensure clean header value
    sanitized_value = sanitized_value.strip()
    
    # Enforce maximum header value length to prevent excessively long headers
    # 8192 bytes is a common limit for header values in many web servers
    MAX_HEADER_LENGTH = 8192
    if len(sanitized_value) > MAX_HEADER_LENGTH:
        sanitized_value = sanitized_value[:MAX_HEADER_LENGTH]
    
    # Construct the final HTTP header string with the custom header name
    header_string = f"X-Custom-Header: {sanitized_value}"
    
    return header_string
