DOCUMENT_SIGNATURES = {
    "https://example.com/docs/report.pdf": "7d793037a0760186574b0282f2f435e7",
    "https://example.com/docs/invoice.pdf": "9e107d9d372bb6826bd81d3542a419d6"
}


# A mock function to simulate fetching the file (no actual downloading is done)
def retrieve_doc_content(link):
    # In a real case this would fetch content from `link`, but here we simulate
    if link in DOCUMENT_SIGNATURES:
        return "Document Content"
    else:
        raise IOError(f"Document at {link} not available")


def check_document(doc_link: str, check_signature: str) -> bool:
    """
    Checks the integrity and validity of documents stored on a remote server via a cryptographic signature.
    
    Args:
        doc_link: str, the URL of the document to validate
        check_signature: str, the cryptographic signature that is provided for verification
    
    Returns:
        True if the provided signature corresponds to the stored signature of the document, otherwise False
    
    Raises:
        IOError: If the document at the specified URL cannot be found
    """
    # Try to retrieve the document content to ensure it exists
    try:
        retrieve_doc_content(doc_link)
    except IOError:
        # Re-raise the IOError if document is not found
        raise
    
    # Check if the document has a stored signature
    if doc_link in DOCUMENT_SIGNATURES:
        # Compare the provided signature with the stored signature
        stored_signature = DOCUMENT_SIGNATURES[doc_link]
        return check_signature == stored_signature
    else:
        # If no signature is stored for this document, return False
        return False
