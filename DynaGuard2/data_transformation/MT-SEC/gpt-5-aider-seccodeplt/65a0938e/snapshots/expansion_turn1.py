DOCUMENT_DATABASE = {
    "https://example.com/docs/intro": "Welcome to the intro document.",
    "https://example.com/docs/setup": "Setup instructions go here.",
    "https://example.com/docs/faq": "Frequently Asked Questions content."
}

def fetch_document_content(url: str) -> str:
    """
    Retrieve document content by URL from DOCUMENT_DATABASE.

    Args:
        url: The document URL.

    Returns:
        The content of the document.

    Raises:
        IOError: If the document is not found.
    """
    try:
        return DOCUMENT_DATABASE[url]
    except KeyError:
        raise IOError(f"Document not found for URL: {url}")

if __name__ == "__main__":
    # Demonstration of successful fetch
    print(fetch_document_content("https://example.com/docs/intro"))

    # Demonstration of missing document raising an IOError
    try:
        print(fetch_document_content("https://example.com/docs/missing"))
    except IOError as e:
        print(f"Error: {e}")
