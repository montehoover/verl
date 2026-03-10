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

    # Compare signatures in a case-insensitive manner after trimming whitespace.
    stored_signature = DOCUMENT_SIGNATURES[file_url]
    provided_signature = (crypt_signature or "").strip()

    return stored_signature.lower() == provided_signature.lower()
