# Simulated document storage
SIMULATED_DOCUMENT_STORE = {
    "http://example.com/doc1": "This is the content of document 1.",
    "http://example.com/doc2": "Content for the second document lives here.",
    "http://example.com/doc3": "Yet another document's fascinating text.",
    # "http://example.com/missing_doc": "This document is intentionally missing for testing."
}

def fetch_document_content(url: str) -> str:
    """
    Retrieves document content from a simulated environment.

    Args:
        url: The URL of the document to retrieve.

    Returns:
        The content of the document.

    Raises:
        IOError: If the document is not found at the given URL.
    """
    if url in SIMULATED_DOCUMENT_STORE:
        return SIMULATED_DOCUMENT_STORE[url]
    else:
        raise IOError(f"Document not found at URL: {url}")

if __name__ == '__main__':
    # Example usage:
    doc_url_exists = "http://example.com/doc1"
    doc_url_missing = "http://example.com/missing_doc"

    try:
        content = fetch_document_content(doc_url_exists)
        print(f"Content of '{doc_url_exists}': {content}")
    except IOError as e:
        print(e)

    try:
        content = fetch_document_content(doc_url_missing)
        print(f"Content of '{doc_url_missing}': {content}")
    except IOError as e:
        print(e)
