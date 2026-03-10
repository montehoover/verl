import requests

def document_exists(doc_link):
    """Check if a document is available at a given URL.
    
    Args:
        doc_link (str): The URL of the document to check.
        
    Returns:
        bool: True if the document is found, False otherwise.
        
    Raises:
        IOError: If the document cannot be accessed.
    """
    try:
        response = requests.get(doc_link, timeout=10)
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            raise IOError(f"Unable to access document: HTTP {response.status_code}")
    except requests.RequestException as e:
        raise IOError(f"Unable to access document: {str(e)}")
