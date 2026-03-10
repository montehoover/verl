import requests

def read_text_file(url: str) -> str:
    """
    Sends a GET request to a given URL and returns the response text.

    Args:
        url: The URL to fetch data from.

    Returns:
        The response text as a string, or an empty string if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.text
    except requests.exceptions.RequestException:
        return ""
