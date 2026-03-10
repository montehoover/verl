DOCUMENT_SIGNATURES = {
    "https://example.com/docs/report.pdf": "7d793037a0760186574b0282f2f435e7",
    "https://example.com/docs/invoice.pdf": "9e107d9d372bb6826bd81d3542a419d6"
}


# A mock function to simulate fetching the file (no actual downloading is done)
def retrieve_document_content(url):
    # In a real case this would fetch content from `url`, but here we simulate
    if url in DOCUMENT_SIGNATURES:
        return "Document Content"
    else:
        raise IOError(f"Document at {url} not available")


def check_signature(file_url: str, crypt_signature: str) -> bool:
    """
    Pure function to check if a cryptographic signature matches the stored signature for a given URL.
    
    Args:
        file_url: str, the URL of the document
        crypt_signature: str, the cryptographic signature to verify
    
    Returns:
        True if the signature matches, False otherwise
    """
    if file_url in DOCUMENT_SIGNATURES:
        return DOCUMENT_SIGNATURES[file_url] == crypt_signature
    return False


def validate_document(file_url: str, crypt_signature: str) -> bool:
    """
    Checks the integrity and validity of documents stored on a remote server via a cryptographic signature.
    
    Args:
        file_url: str, the URL of the document to validate
        crypt_signature: str, the cryptographic signature that is provided for verification
    
    Returns:
        True if the provided signature corresponds to the stored signature of the document, otherwise False
    
    Raises:
        IOError: If the document at the specified URL cannot be found
    """
    # Try to retrieve the document content (will raise IOError if not found)
    retrieve_document_content(file_url)
    
    # Check the signature using the pure function
    return check_signature(file_url, crypt_signature)
