from urllib.parse import urlparse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KNOWN_DOMAINS = ["trusted.source.com", "callback.handler.net", "example.com", "sub.example.co.uk"]

def validate_callback(callback_url: str) -> str:
    """
    Parses, validates, and logs the callback URL.
    Checks if the domain is known and logs an alert for unknown domains.

    Args:
        callback_url: The URL string from the webhook payload.

    Returns:
        The parsed URL as a string.
    """
    parsed_url = urlparse(callback_url)
    domain = parsed_url.hostname

    if domain and domain not in KNOWN_DOMAINS:
        logging.warning(f"ALERT: Unknown domain detected: {domain} in URL: {parsed_url.geturl()}")
    else:
        logging.info(f"Domain {domain} is known or no domain present.")

    logging.info(f"Parsed callback URL: {parsed_url.geturl()}")
    return parsed_url.geturl()

if __name__ == '__main__':
    # Example usage:
    test_url = "http://example.com/callback?param1=value1&param2=value2" # Known domain
    validated_url = validate_callback(test_url)
    print(f"Validated URL returned: {validated_url}")

    test_url_https = "https://sub.example.co.uk:8080/path/to/resource?query=string#fragment" # Known domain
    validated_url_https = validate_callback(test_url_https)
    print(f"Validated URL returned: {validated_url_https}")

    test_unknown_domain_url = "http://unknown.domain.com/some/path" # Unknown domain
    validated_unknown_domain_url = validate_callback(test_unknown_domain_url)
    print(f"Validated URL returned: {validated_unknown_domain_url}")

    test_trusted_url = "https://trusted.source.com/api/v1/update" # Known domain
    validated_trusted_url = validate_callback(test_trusted_url)
    print(f"Validated URL returned: {validated_trusted_url}")

    test_invalid_url = "not_a_url" # Example of a string that urlparse will still process (no domain)
    validated_invalid_url = validate_callback(test_invalid_url)
    print(f"Validated URL returned: {validated_invalid_url}")
