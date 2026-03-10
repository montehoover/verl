import requests

def document_exists(doc_link: str) -> bool:
    """
    Checks if a document is available at a given URL.

    Args:
        doc_link: The URL of the document.

    Returns:
        True if the document is found (HTTP 200), False if not found (HTTP 404).

    Raises:
        IOError: If the document cannot be accessed due to network issues
                 or other HTTP errors.
    """
    try:
        response = requests.head(doc_link, timeout=10)  # Use HEAD for efficiency
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            raise IOError(f"Error accessing document at {doc_link}: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        raise IOError(f"Error accessing document at {doc_link}: {e}")
