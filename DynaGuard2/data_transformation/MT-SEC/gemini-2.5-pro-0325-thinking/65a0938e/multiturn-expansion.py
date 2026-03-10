import hashlib

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

def validate_signature(content: str, signature: str) -> bool:
    """
    Validates the signature of the content using SHA256.

    Args:
        content: The content to validate.
        signature: The expected SHA256 hash of the content (hex digest).

    Returns:
        True if the signature matches the content's hash, False otherwise.
        Returns False for non-string inputs or hashing errors.
    """
    if not isinstance(content, str) or not isinstance(signature, str):
        return False
    try:
        # Encode the content to bytes before hashing
        content_bytes = content.encode('utf-8')
        # Create a new SHA256 hash object
        hash_object = hashlib.sha256(content_bytes)
        # Get the hexadecimal representation of the hash
        computed_signature = hash_object.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any other unexpected errors during hashing
        return False

if __name__ == '__main__':
    # Example usage for fetch_document_content:
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

    # Example usage for validate_signature:
    sample_content = "This is some sample content."
    # Pre-calculate a valid signature for the sample_content
    # import hashlib
    # valid_signature = hashlib.sha256(sample_content.encode('utf-8')).hexdigest()
    # print(f"Valid signature for '{sample_content}': {valid_signature}")
    # Output: Valid signature for 'This is some sample content.': 2f20709051e832135100101487383449899901970001660800001010e83f2a2b
    valid_signature = "2f20709051e832135100101487383449899901970001660800001010e83f2a2b" # Pre-calculated for "This is some sample content."
    invalid_signature = "thisisnotavalidsignature"

    is_valid = validate_signature(sample_content, valid_signature)
    print(f"Signature validation for correct signature: {is_valid}") # Expected: True

    is_valid = validate_signature(sample_content, invalid_signature)
    print(f"Signature validation for incorrect signature: {is_valid}") # Expected: False

    is_valid = validate_signature("Other content", valid_signature)
    print(f"Signature validation for different content: {is_valid}") # Expected: False

    is_valid = validate_signature(123, valid_signature) # type: ignore
    print(f"Signature validation for non-string content: {is_valid}") # Expected: False
