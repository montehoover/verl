from urllib.parse import urlparse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_callback(callback_url: str) -> str:
    """
    Validates a webhook callback URL against a whitelist of approved domains
    and checks for attempts to access internal network resources.

    Args:
        callback_url: The URL string from the webhook payload.

    Returns:
        The validated and secure URL as a string.

    Raises:
        ValueError: If the URL's domain is not in APPROVED_API_DOMAINS,
                    if the URL attempts to point to a private network address,
                    or if the URL is malformed.
    """
    if not callback_url or not isinstance(callback_url, str):
        raise ValueError("Callback URL must be a non-empty string.")

    try:
        parsed_url = urlparse(callback_url)
    except Exception as e: # Catch any parsing errors, though urlparse is quite robust
        logging.error(f"Error parsing URL '{callback_url}': {e}")
        raise ValueError(f"Malformed callback URL: {callback_url}") from e

    domain = parsed_url.hostname
    scheme = parsed_url.scheme

    if not domain or not scheme: # Basic check for a valid URL structure
        raise ValueError(f"Invalid URL structure: {callback_url}. Scheme and hostname are required.")

    # Check 1: Domain must be in the approved list
    if domain not in APPROVED_API_DOMAINS:
        logging.error(f"Domain validation failed: {domain} is not an approved API domain.")
        raise ValueError(f"Domain '{domain}' is not an approved API endpoint.")

    # Check 2: Prevent access to internal networks
    # This checks if the hostname itself is an internal IP or starts with an internal range prefix.
    # For more robust IP validation, consider using the `ipaddress` module if available/allowed.
    if any(domain.startswith(internal_ip) for internal_ip in INTERNAL_IP_RANGES):
        logging.error(f"Internal network access attempt: {domain} matches an internal IP range.")
        raise ValueError(f"URL '{callback_url}' points to a restricted internal network address.")

    # Check 3: Ensure HTTPS for security (optional, but good practice)
    if scheme.lower() != 'https':
        logging.warning(f"URL '{callback_url}' uses non-HTTPS scheme: {scheme}. Consider enforcing HTTPS.")
        # Depending on strictness, you might raise ValueError here too:
        # raise ValueError("Callback URL must use HTTPS.")

    logging.info(f"Successfully validated callback URL: {parsed_url.geturl()}")
    return parsed_url.geturl()

if __name__ == '__main__':
    test_urls = [
        "https://api.service.com/data/update",  # Valid
        "http://hooks.webapp.com/event",       # Valid (but http, will log warning)
        "https://another.api.com/resource",    # Invalid domain
        "https://10.0.0.5/internal/service",   # Internal IP
        "https://localhost:8080/callback",     # Localhost
        "ftp://api.service.com/files",         # Invalid scheme (if HTTPS enforced strictly)
        "not_a_valid_url",                     # Malformed
        "https://192.168.1.100/mydevice",      # Internal IP
        "https://hooks.webapp.com",            # Valid
        "",                                    # Empty URL
        None,                                  # None URL
    ]

    for url in test_urls:
        print(f"\nTesting URL: {url}")
        try:
            validated_url = validate_callback(url)
            print(f"Validation successful: {validated_url}")
        except ValueError as e:
            print(f"Validation failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
