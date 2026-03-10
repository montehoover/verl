from datetime import datetime
import logging
from urllib.parse import urlparse

ALLOWED_SCHEMES = ["https"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def enforce_api_security(api_endpoint: str) -> str:
    """
    Ensures API endpoint uses 'https' and logs access.

    Args:
        api_endpoint: The URL of the API endpoint being accessed.

    Returns:
        The original api_endpoint string if secure.

    Raises:
        ValueError: If the api_endpoint does not use the 'https' scheme.
    """
    parsed_url = urlparse(api_endpoint)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"API endpoint '{api_endpoint}' must use 'https' scheme. Found: '{parsed_url.scheme}'"
        )

    # Log the safe endpoint access
    logging.info(f"Secure API endpoint accessed: {api_endpoint}")

    # Return the original URL string for further processing
    return api_endpoint

if __name__ == '__main__':
    # Example Usage:
    print("Testing with a valid HTTPS endpoint:")
    valid_url = "https://api.example.com/data"
    try:
        returned_url = enforce_api_security(valid_url)
        print(f"Function returned: {returned_url}\n")
    except ValueError as e:
        print(f"Caught unexpected error: {e}\n")


    print("Testing with an endpoint missing scheme (should raise ValueError):")
    invalid_url_no_scheme = "api.example.com/user"
    try:
        returned_url = enforce_api_security(invalid_url_no_scheme)
        print(f"Function returned: {returned_url}\n")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}\n")

    print("Testing with an FTP endpoint (should raise ValueError):")
    ftp_url = "ftp://api.example.com/files"
    try:
        returned_url = enforce_api_security(ftp_url)
        print(f"Function returned: {returned_url}\n")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}\n")

    print("Testing with an HTTP endpoint (should raise ValueError):")
    http_url = "http://api.example.com/status"
    try:
        returned_url = enforce_api_security(http_url)
        print(f"Function returned: {returned_url}\n")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}\n")

    print("Testing with another valid HTTPS endpoint:")
    valid_https_url_again = "https://secure.anotherexample.com/api/v1/resource"
    try:
        returned_url = enforce_api_security(valid_https_url_again)
        print(f"Function returned: {returned_url}\n")
    except ValueError as e:
        print(f"Caught unexpected error: {e}\n")
