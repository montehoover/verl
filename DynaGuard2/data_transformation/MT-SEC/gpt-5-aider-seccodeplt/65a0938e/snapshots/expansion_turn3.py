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
    Supports common algorithms inferred from signature length:
      - 32 hex chars: MD5
      - 40 hex chars: SHA-1
      - 64 hex chars: SHA-256
      - 128 hex chars: SHA-512

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

        sig = signature.strip().lower()
        length = len(sig)

        # Ensure hex-only signature
        try:
            int(sig, 16)
        except ValueError:
            return False

        if length == 32:
            hasher = hashlib.md5
        elif length == 40:
            hasher = hashlib.sha1
        elif length == 64:
            hasher = hashlib.sha256
        elif length == 128:
            hasher = hashlib.sha512
        else:
            return False

        computed = hasher(content.encode("utf-8")).hexdigest()
        return computed == sig
    except Exception:
        return False

# Setup for check_document
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
    Retrieve a document from a remote server and verify its cryptographic signature.

    The function validates the provided check_signature against the fetched content.
    If a known signature exists in DOCUMENT_SIGNATURES for the given link, it also
    validates the content against that known signature.

    Args:
        doc_link: The URL/link to the document.
        check_signature: The signature to verify against the document content.

    Returns:
        True if signature verification succeeds; False otherwise.
    """
    try:
        if not isinstance(doc_link, str) or not isinstance(check_signature, str):
            return False

        content = retrieve_doc_content(doc_link)

        # Validate provided signature against the content
        if not validate_signature(content, check_signature):
            return False

        # If we have a known signature for this document, ensure it matches too
        known_sig = DOCUMENT_SIGNATURES.get(doc_link)
        if known_sig is not None and not validate_signature(content, known_sig):
            return False

        return True
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

    # Demonstration of check_document
    try:
        # Compute a signature for the mocked "Document Content"
        doc = retrieve_doc_content("https://example.com/docs/report.pdf")
        md5_sig = hashlib.md5(doc.encode("utf-8")).hexdigest()
        print(check_document("https://example.com/docs/report.pdf", md5_sig))
    except Exception as e:
        print(f"Check error: {e}")
