import hashlib

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

def validate_signature(content: str, signature: str) -> bool:
    """
    Validate a signature by hashing the content and comparing it to the provided signature.
    Uses SHA-256 for hashing.

    Args:
        content: The content to hash.
        signature: The expected hex digest string.

    Returns:
        True if the signature matches the computed hash; False otherwise.
        Returns False on unsupported inputs or any failure.
    """
    try:
        if not isinstance(content, str) or not isinstance(signature, str):
            return False
        computed = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return computed == signature.strip().lower()
    except Exception:
        return False

if __name__ == "__main__":
    # Demonstration of successful fetch
    print(fetch_document_content("https://example.com/docs/intro"))

    # Demonstration of missing document raising an IOError
    try:
        print(fetch_document_content("https://example.com/docs/missing"))
    except IOError as e:
        print(f"Error: {e}")

    # Demonstration of signature validation
    try:
        content = fetch_document_content("https://example.com/docs/intro")
        good_sig = hashlib.sha256(content.encode("utf-8")).hexdigest()
        bad_sig = "deadbeef"
        print(validate_signature(content, good_sig))  # Expected True
        print(validate_signature(content, bad_sig))   # Expected False
    except Exception as e:
        print(f"Validation error: {e}")
