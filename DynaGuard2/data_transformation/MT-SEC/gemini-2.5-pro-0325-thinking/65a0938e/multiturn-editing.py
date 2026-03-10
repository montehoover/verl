# A predefined list of known document URLs
KNOWN_DOCUMENT_URLS = [
    "http://example.com/doc1",
    "http://example.com/doc2",
    "https://example.org/documentA",
]

def document_exists(doc_link: str) -> bool:
    """
    Checks if a document URL is in a predefined list of known document URLs.

    Args:
        doc_link: The URL of the document to check.

    Returns:
        True if the doc_link is in the predefined list, False otherwise.
    """
    return doc_link in KNOWN_DOCUMENT_URLS
