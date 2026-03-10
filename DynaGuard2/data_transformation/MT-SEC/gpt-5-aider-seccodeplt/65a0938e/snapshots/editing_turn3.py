from typing import Dict

DOCUMENT_SIGNATURES: Dict[str, str] = {
    "https://example.com/docs/report.pdf": "7d793037a0760186574b0282f2f435e7",
    "https://example.com/docs/invoice.pdf": "9e107d9d372bb6826bd81d3542a419d6",
}


def retrieve_doc_content(link: str) -> str:
    if link in DOCUMENT_SIGNATURES:
        return "Document Content"
    else:
        raise IOError(f"Document at {link} not available")


def check_document(doc_link: str, check_signature: str) -> bool:
    """
    Verify the integrity of a document by comparing the provided signature
    against the stored signature for the given document URL.

    Args:
        doc_link: The URL of the document to verify.
        check_signature: The cryptographic signature to check.

    Returns:
        True if the provided signature matches the stored signature, otherwise False.

    Raises:
        IOError: If the document at the specified URL cannot be found.
    """
    try:
        # Ensure the document exists (raises IOError if not available)
        retrieve_doc_content(doc_link)
    except IOError:
        # Propagate as per the requirements
        raise

    expected_signature = DOCUMENT_SIGNATURES.get(doc_link)
    if expected_signature is None:
        # Redundant safeguard; retrieve_doc_content should have raised already.
        raise IOError(f"Document at {doc_link} not available")

    return check_signature == expected_signature
