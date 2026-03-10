from typing import Dict
import hmac

DOCUMENT_SIGNATURES: Dict[str, str] = {
    "https://example.com/docs/report.pdf": "7d793037a0760186574b0282f2f435e7",
    "https://example.com/docs/invoice.pdf": "9e107d9d372bb6826bd81d3542a419d6"
}


# A mock function to simulate fetching the file (no actual downloading is done)
def retrieve_doc_content(link: str) -> str:
    # In a real case this would fetch content from `link`, but here we simulate
    if link in DOCUMENT_SIGNATURES:
        return "Document Content"
    else:
        raise IOError(f"Document at {link} not available")


def check_document(doc_link: str, check_signature: str) -> bool:
    """
    Check the integrity and validity of a remote document using a cryptographic signature.

    Args:
        doc_link: The URL of the document to validate.
        check_signature: The cryptographic signature provided for verification.

    Returns:
        True if the provided signature matches the stored signature for the document,
        otherwise False.

    Raises:
        IOError: If the document at the specified URL cannot be found.
    """
    # Ensure the document exists/accessible. This will raise IOError if not available.
    retrieve_doc_content(doc_link)

    # Get the stored signature for the document, if any
    stored_signature = DOCUMENT_SIGNATURES.get(doc_link)
    if stored_signature is None:
        # No known signature for this document; cannot validate positively.
        return False

    # Perform a constant-time comparison after normalizing case/whitespace
    provided = check_signature.strip().lower()
    expected = stored_signature.strip().lower()
    return hmac.compare_digest(provided, expected)
