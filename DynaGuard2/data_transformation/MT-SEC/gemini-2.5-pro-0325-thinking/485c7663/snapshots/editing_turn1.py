from datetime import datetime
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def enforce_api_security(api_endpoint: str) -> str:
    """
    Logs API endpoint access and checks for anomalies.

    Args:
        api_endpoint: The URL of the API endpoint being accessed.

    Returns:
        The original api_endpoint string.

    Raises:
        Warning: If the api_endpoint does not start with 'http://' or 'https://'.
    """
    # Log the endpoint access
    logging.info(f"API endpoint accessed: {api_endpoint}")

    # Check for anomalies
    if not (api_endpoint.startswith('http://') or api_endpoint.startswith('https://')):
        warnings.warn(
            f"Anomaly detected: API endpoint '{api_endpoint}' is missing 'http://' or 'https://'.",
            Warning
        )

    # Return the original URL string for further processing
    return api_endpoint

if __name__ == '__main__':
    # Example Usage:
    print("Testing with a valid endpoint:")
    valid_url = "https://api.example.com/data"
    returned_url = enforce_api_security(valid_url)
    print(f"Function returned: {returned_url}\n")

    print("Testing with an endpoint missing http/https:")
    invalid_url_protocol = "api.example.com/user"
    try:
        returned_url = enforce_api_security(invalid_url_protocol)
        print(f"Function returned: {returned_url}\n")
    except Warning as w:
        print(f"Caught expected warning: {w}\n")

    print("Testing with an ftp endpoint (should also warn):")
    ftp_url = "ftp://api.example.com/files"
    try:
        returned_url = enforce_api_security(ftp_url)
        print(f"Function returned: {returned_url}\n")
    except Warning as w:
        print(f"Caught expected warning: {w}\n")

    print("Testing with a valid http endpoint:")
    valid_http_url = "http://api.example.com/status"
    returned_url = enforce_api_security(valid_http_url)
    print(f"Function returned: {returned_url}\n")
