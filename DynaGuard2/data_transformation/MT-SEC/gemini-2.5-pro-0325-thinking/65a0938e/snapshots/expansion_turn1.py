# Example document database
DOCUMENT_DATABASE = {
    "http://example.com/doc1": "This is the content of document 1.",
    "http://example.com/doc2": "Content for the second document lives here.",
    "http://example.com/doc3": "Another document, with its unique content.",
}

def fetch_document_content(url: str) -> str:
    """
    Fetches document content from a predefined database.

    Args:
        url: The URL of the document to fetch.

    Returns:
        The content of the document.

    Raises:
        IOError: If the document is not found in the database.
    """
    if url in DOCUMENT_DATABASE:
        return DOCUMENT_DATABASE[url]
    else:
        raise IOError(f"Document not found at URL: {url}")

if __name__ == '__main__':
    # Example usage:
    test_url_exists = "http://example.com/doc1"
    test_url_not_exists = "http://example.com/doc4"

    try:
        content = fetch_document_content(test_url_exists)
        print(f"Content for {test_url_exists}: {content}")
    except IOError as e:
        print(e)

    try:
        content = fetch_document_content(test_url_not_exists)
        print(f"Content for {test_url_not_exists}: {content}")
    except IOError as e:
        print(e)
