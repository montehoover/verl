import re

def add_safe_header(custom_value: str) -> str:
    """
    Inserts a user-supplied value as a custom header in the HTTP response headers.
    The value must comply with valid header rules and should prevent potential security risks.

    Args:
        custom_value: str, the user-provided custom header value.

    Returns:
        str, representing the final header string including the sanitized custom header value.
    
    Raises:
        Does not raise any specific exceptions, but improper input could indirectly raise 're.error'.
    """
    # Sanitize the input value to make it safe for use in an HTTP header.
    
    # Step 1: Remove characters not typically allowed or safe in header values.
    # The regex r'[^\w\s-]' matches any character that is NOT:
    #   \w (alphanumeric characters [a-zA-Z0-9] plus underscore _),
    #   \s (whitespace characters like space, tab, newline), or
    #   - (hyphen).
    # These disallowed characters are removed by replacing them with an empty string.
    # Note: \s includes newlines at this stage, which are handled in the next step.
    sanitized_value = re.sub(r'[^\w\s-]', '', custom_value)
    
    # Step 2: Explicitly remove any newline (\n) or carriage return (\r) characters.
    # The regex r'[\r\n]' matches either a carriage return or a newline.
    # This is a crucial security measure to prevent CRLF injection (HTTP Header Injection)
    # vulnerabilities. Injected newlines could otherwise be used by an attacker to
    # terminate the current header and inject new, malicious headers or even a fake response body.
    sanitized_value = re.sub(r'[\r\n]', '', sanitized_value)
    
    # Step 3: Trim leading and trailing whitespace from the sanitized value.
    # While Steps 1 and 2 might leave legitimate spaces or tabs (e.g., " value "),
    # leading/trailing whitespace in header values can be inconsistent across systems
    # or simply undesirable. Stripping ensures a cleaner, more predictable value.
    sanitized_value = sanitized_value.strip()
    
    # Construct the final header string.
    # "X-Custom-Header" uses the conventional "X-" prefix for non-standard, custom HTTP headers.
    header_name = "X-Custom-Header"
    return f"{header_name}: {sanitized_value}"
