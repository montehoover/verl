import requests

# Predefined list of valid document URLs
VALID_DOCUMENT_LINKS = [
    "https://example.com/docs/doc1.pdf",
    "https://example.com/docs/doc2.pdf",
    "https://example.com/docs/manual.pdf",
    "https://example.com/docs/guide.pdf",
    "https://example.com/docs/reference.pdf"
]

def document_exists(doc_link):
    """Check if a document URL is in the predefined list of valid document links.
    
    Args:
        doc_link (str): The URL of the document to check.
        
    Returns:
        bool: True if the URL is in the predefined list, False otherwise.
    """
    return doc_link in VALID_DOCUMENT_LINKS
