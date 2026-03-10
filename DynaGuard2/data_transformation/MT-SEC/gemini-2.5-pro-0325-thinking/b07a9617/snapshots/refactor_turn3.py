import re
import logging

# Configure a basic logger
# In a real application, this configuration might be more complex and centralized.
logger = logging.getLogger(__name__)
# Set a default logging level if no handlers are configured.
# This is useful for scripts/modules; applications would typically configure logging.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
    logger.debug(f"Initial custom_header_value: '{custom_header_value}'")

    # Step 1: Remove carriage returns (CR) and line feeds (LF).
    # This is crucial to prevent HTTP Response Splitting attacks,
    # where an attacker might inject CR/LF characters to add fake
    # headers or even a fake body.
    value_after_crlf_removal = re.sub(r'[\r\n]', '', custom_header_value)
    if value_after_crlf_removal != custom_header_value:
        logger.debug(f"Value after CR/LF removal: '{value_after_crlf_removal}'")
    else:
        logger.debug("No CR/LF characters found to remove.")
    
    sanitized_value = value_after_crlf_removal

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
    value_after_char_restriction = re.sub(r'[^a-zA-Z0-9_ .!#$%&\'*+\-/^`|~]+', '', sanitized_value)
    if value_after_char_restriction != sanitized_value:
        logger.debug(f"Value after character restriction: '{value_after_char_restriction}'")
    else:
        logger.debug("No characters removed by restriction filter.")
    
    sanitized_value = value_after_char_restriction
    
    # Step 3: Remove leading and trailing whitespace.
    # While many servers or HTTP libraries might trim whitespace from
    # header values automatically, explicitly stripping it ensures
    # consistent behavior and aligns with best practices for
    # well-formed header values.
    value_after_stripping = sanitized_value.strip()
    if value_after_stripping != sanitized_value:
        logger.debug(f"Value after stripping whitespace: '{value_after_stripping}'")
    else:
        logger.debug("No leading/trailing whitespace to strip.")

    sanitized_value = value_after_stripping

    final_header = f"X-Custom-Header: {sanitized_value}"
    logger.debug(f"Final sanitized header: '{final_header}'")
    return final_header
