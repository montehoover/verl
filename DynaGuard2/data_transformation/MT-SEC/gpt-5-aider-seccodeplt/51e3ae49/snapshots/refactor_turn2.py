import hmac

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


def _signatures_match(provided_signature: str, expected_signature: str) -> bool:
    """
    Pure function to check if the provided signature matches the expected signature.

    Normalizes both signatures (trim + case-insensitive) and performs a
    constant-time comparison to mitigate timing attacks.

    Args:
        provided_signature: The signature to verify.
        expected_signature: The known, correct signature.

    Returns:
        True if they match; otherwise False.
    """
    # Normalize inputs: trim whitespace and compare case-insensitively (hex-friendly)
    provided_norm = (provided_signature or "").strip().lower()
    expected_norm = (expected_signature or "").strip().lower()
    return hmac.compare_digest(provided_norm, expected_norm)


def validate_document(file_url: str, crypt_signature: str) -> bool:
    """
    Verify that the provided cryptographic signature matches the stored signature
    for the document at the given URL. Verification happens before any content
    access or processing.

    Args:
        file_url: The URL of the document to validate.
        crypt_signature: The cryptographic signature to verify.

    Returns:
        True if the provided signature matches the stored signature; otherwise False.

    Raises:
        IOError: If the document at the specified URL cannot be found.
    """
    # Ensure the document exists; if not, raise as required.
    if file_url not in DOCUMENT_SIGNATURES:
        raise IOError(f"Document at {file_url} not available")

    # Retrieve the stored signature and compare using the pure helper function.
    stored_signature = DOCUMENT_SIGNATURES[file_url]
    return _signatures_match(crypt_signature, stored_signature)
