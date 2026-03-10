import requests

def read_text_file(url: str) -> str:
    """
    Send a GET request to the given URL and return the response text.

    Args:
        url: The URL to request.

    Returns:
        The response body as text, or an empty string if the request fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return ""
