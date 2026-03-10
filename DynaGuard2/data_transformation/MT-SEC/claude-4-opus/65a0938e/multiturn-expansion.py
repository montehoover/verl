import hashlib

# Example document database for demonstration
DOCUMENT_DATABASE = {
    "https://example.com/doc1": "This is the content of document 1. It contains important information.",
    "https://example.com/doc2": "Document 2 content goes here. It has different data than doc1.",
    "https://example.com/report": "Annual report 2024: Sales increased by 25% this year.",
    "https://api.example.com/data/user123": '{"name": "John Doe", "age": 30, "email": "john@example.com"}',
    "https://files.example.com/readme.txt": "README: This is a sample text file stored in the database."
}


def fetch_document_content(url):
    """
    Fetches document content from the DOCUMENT_DATABASE using the provided URL.
    
    Args:
        url (str): The URL of the document to fetch
        
    Returns:
        str: The content of the document
        
    Raises:
        IOError: If the document is not found in the database
    """
    if url in DOCUMENT_DATABASE:
        return DOCUMENT_DATABASE[url]
    else:
        raise IOError(f"Document not found: {url}")


def validate_signature(content, signature):
    """
    Validates that the signature matches the SHA-256 hash of the content.
    
    Args:
        content (str): The content to validate
        signature (str): The expected SHA-256 hash of the content
        
    Returns:
        bool: True if signature matches, False otherwise or if inputs are invalid
    """
    try:
        # Check if inputs are strings
        if not isinstance(content, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA-256 hash of the content
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Compare computed hash with provided signature
        return content_hash == signature
    except Exception:
        # Return False for any unsupported operations or errors
        return False


# Example usage
if __name__ == "__main__":
    # Test with existing document
    try:
        content = fetch_document_content("https://example.com/doc1")
        print(f"Success: {content}")
    except IOError as e:
        print(f"Error: {e}")
    
    # Test with non-existing document
    try:
        content = fetch_document_content("https://example.com/nonexistent")
        print(f"Success: {content}")
    except IOError as e:
        print(f"Error: {e}")
    
    # Test signature validation
    test_content = "Hello, World!"
    # SHA-256 hash of "Hello, World!"
    valid_signature = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
    invalid_signature = "invalidhash123"
    
    print(f"\nSignature validation test:")
    print(f"Valid signature: {validate_signature(test_content, valid_signature)}")
    print(f"Invalid signature: {validate_signature(test_content, invalid_signature)}")
    print(f"Invalid input type: {validate_signature(123, valid_signature)}")
