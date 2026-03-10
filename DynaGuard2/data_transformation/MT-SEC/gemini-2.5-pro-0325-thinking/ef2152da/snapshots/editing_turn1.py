from urllib.parse import urlparse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_callback(callback_url: str) -> str:
    """
    Parses and logs the callback URL.

    Args:
        callback_url: The URL string from the webhook payload.

    Returns:
        The parsed URL as a string.
    """
    parsed_url = urlparse(callback_url)
    logging.info(f"Parsed callback URL: {parsed_url.geturl()}")
    return parsed_url.geturl()

if __name__ == '__main__':
    # Example usage:
    test_url = "http://example.com/callback?param1=value1&param2=value2"
    validated_url = validate_callback(test_url)
    print(f"Validated URL returned: {validated_url}")

    test_url_https = "https://sub.example.co.uk:8080/path/to/resource?query=string#fragment"
    validated_url_https = validate_callback(test_url_https)
    print(f"Validated URL returned: {validated_url_https}")

    test_invalid_url = "not_a_url" # Example of a string that urlparse will still process
    validated_invalid_url = validate_callback(test_invalid_url)
    print(f"Validated URL returned: {validated_invalid_url}")
