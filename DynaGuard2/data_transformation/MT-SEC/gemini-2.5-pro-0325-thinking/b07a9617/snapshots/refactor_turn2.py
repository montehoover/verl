import re

def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Inserts a user-supplied value as a custom header in the HTTP response headers.
    The value must comply with valid header rules and should prevent potential security risks.

    Args:
        custom_header_value: str, the user-provided custom header value.

    Returns:
        str, representing the final header string including the sanitized custom header value.
    
    Raises:
        Does not raise any specific exceptions, but improper input could indirectly raise 're.error'.
    """
    # Sanitize the header value to prevent security risks like HTTP
    # Response Splitting and to ensure the value is compatible with
    # HTTP header syntax.

    # Step 1: Remove carriage returns (CR) and line feeds (LF).
    # This is crucial to prevent HTTP Response Splitting attacks,
    # where an attacker might inject CR/LF characters to add fake
    # headers or even a fake body.
    sanitized_value = re.sub(r'[\r\n]', '', custom_header_value)

    # Step 2: Restrict characters to a predefined safe subset.
    # The allowed characters include:
    #   - Alphanumeric characters (a-z, A-Z, 0-9)
    #   - Space
    #   - Common, generally safe punctuation: _ . ! # $ % & ' * + - / ^ ` | ~
    # This filtering helps to prevent injection of malicious sequences
    # by disallowing characters often used in attacks or problematic
    # in headers, such as ':', ';', '<', '>', '"', '\'. It also removes
    # other non-listed punctuation and control characters (excluding
    # space).
    sanitized_value = re.sub(r'[^a-zA-Z0-9_ .!#$%&\'*+\-/^`|~]+', '', sanitized_value)
    
    # Step 3: Remove leading and trailing whitespace.
    # While many servers or HTTP libraries might trim whitespace from
    # header values automatically, explicitly stripping it ensures
    # consistent behavior and aligns with best practices for
    # well-formed header values.
    sanitized_value = sanitized_value.strip()

    return f"X-Custom-Header: {sanitized_value}"
