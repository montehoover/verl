import hashlib

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

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates if the provided signature matches the data's SHA256 hash.

    Args:
        data: The data string to validate.
        signature: The expected SHA256 signature (hex digest) of the data.

    Returns:
        True if the signature matches the data, False otherwise.
        Returns False for non-string inputs or other errors.
    """
    if not isinstance(data, str) or not isinstance(signature, str):
        return False
    try:
        # Hash the data using SHA256
        hashed_data = hashlib.sha256(data.encode('utf-8')).hexdigest()
        # Compare the calculated hash with the provided signature
        return hashed_data == signature
    except Exception:
        # Catch any other unexpected errors during hashing or comparison
        return False

if __name__ == '__main__':
    # Example usage for fetch_document_content:
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

    print("\n--- Signature Validation Examples ---")
    sample_data = "This is some important data."
    # Generate a correct signature for the sample_data
    correct_signature = hashlib.sha256(sample_data.encode('utf-8')).hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    # Test with correct signature
    is_valid = validate_signature(sample_data, correct_signature)
    print(f"Validation for correct signature: {is_valid}") # Expected: True

    # Test with incorrect signature
    is_valid = validate_signature(sample_data, incorrect_signature)
    print(f"Validation for incorrect signature: {is_valid}") # Expected: False

    # Test with non-string data
    is_valid = validate_signature(12345, correct_signature) # type: ignore
    print(f"Validation for non-string data: {is_valid}") # Expected: False

    # Test with non-string signature
    is_valid = validate_signature(sample_data, None) # type: ignore
    print(f"Validation for non-string signature: {is_valid}") # Expected: False
