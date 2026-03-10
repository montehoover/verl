# Assuming RESOURCE_AVAILABILITY is defined elsewhere
RESOURCE_AVAILABILITY = {}

# Assuming retrieve_document_content is defined elsewhere
def retrieve_document_content(url):
    # Placeholder implementation
    pass

def fetch_document(file_url):
    """
    Retrieves data from a remote server.
    
    Args:
        file_url (str): The document location URL
        
    Returns:
        The document content if the URL is valid
        
    Raises:
        IOError: If the document cannot be found or connectivity issues occur
    """
    if not isinstance(file_url, str):
        raise IOError("Invalid URL: must be a string")
    
    if not file_url:
        raise IOError("Invalid URL: cannot be empty")
    
    # Check if the resource is available
    if file_url in RESOURCE_AVAILABILITY and not RESOURCE_AVAILABILITY[file_url]:
        raise IOError(f"Connectivity error: Resource at {file_url} is not available")
    
    try:
        content = retrieve_document_content(file_url)
        if content is None:
            raise IOError(f"Document not found at {file_url}")
        return content
    except Exception as e:
        if isinstance(e, IOError):
            raise
        raise IOError(f"Failed to retrieve document from {file_url}: {str(e)}")
