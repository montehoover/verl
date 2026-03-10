from urllib.parse import urlparse

def process_webhook(webhook_url: str) -> dict:
    """
    Parses a webhook URL and returns its components.

    Args:
        webhook_url: The URL string to parse.

    Returns:
        A dictionary containing the scheme, domain, and path of the URL.
    """
    parsed_url = urlparse(webhook_url)
    return {
        "scheme": parsed_url.scheme,
        "domain": parsed_url.hostname,
        "path": parsed_url.path,
    }
