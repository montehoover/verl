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
    Checks the integrity and validity of a remote document via a cryptographic signature.

    Args:
        doc_link: The URL of the document to validate.
        check_signature: The cryptographic signature provided for verification.

    Returns:
        True if the provided signature matches the stored signature of the document,
        otherwise False.

    Raises:
        IOError: If the document at the specified URL cannot be found.
    """
    # Ensure the document exists (simulate "not found")
    if doc_link not in DOCUMENT_SIGNATURES:
        raise IOError(f"Document at {doc_link} not available")

    # Verify signature without accessing/processing the document content
    expected_signature = DOCUMENT_SIGNATURES[doc_link]
    return check_signature == expected_signature
