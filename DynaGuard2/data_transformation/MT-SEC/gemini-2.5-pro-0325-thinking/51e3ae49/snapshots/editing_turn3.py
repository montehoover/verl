# Mock global resource availability
RESOURCE_AVAILABILITY = {
    "http://example.com/doc1": "This is document 1.",
    "http://example.com/doc2": "Contents of document 2.",
    "http://example.com/missing_doc": None, # Represents a document that cannot be found
    "http://example.com/error_doc": "ERROR" # Represents a connectivity issue
}

def retrieve_document_content(file_url: str) -> str:
    """
    Simulates retrieving document content from a remote server.
    - If the URL is in RESOURCE_AVAILABILITY and has content, returns it.
    - If the URL is in RESOURCE_AVAILABILITY but content is None, simulates "not found".
    - If the URL is in RESOURCE_AVAILABILITY and content is "ERROR", simulates a connectivity error.
    - If the URL is not in RESOURCE_AVAILABILITY, it's treated as "not found".
    """
    if file_url in RESOURCE_AVAILABILITY:
        content = RESOURCE_AVAILABILITY[file_url]
        if content == "ERROR":
            # Simulate a generic connectivity problem for this specific function
            # In a real scenario, this might be a network exception from a library like requests
            raise ConnectionError(f"Simulated connection error for {file_url}")
        if content is None:
            # Simulate document not found at the source
            return None # Or raise a specific "NotFound" exception if preferred
        return content
    return None # Document not listed in our mock availability, treat as not found

# Global dictionary of document signatures
DOCUMENT_SIGNATURES = {
    "http://example.com/doc1": "signature_doc1_valid",
    "http://example.com/doc2": "signature_doc2_valid",
    # missing_doc and error_doc do not have signatures here for testing purposes
}

# Predetermined list of known document URLs (using keys from RESOURCE_AVAILABILITY for this example)
KNOWN_DOCUMENT_URLS = list(RESOURCE_AVAILABILITY.keys())

def fetch_document(file_url: str, consider_existence: bool) -> str:
    """
    Retrieves data from a remote server.

    Args:
        file_url: The URL of the document to retrieve.
        consider_existence: If True, checks if the URL is in a predetermined list
                            before attempting retrieval.

    Returns:
        The document content if found. If consider_existence is True and the URL
        is not in the predetermined list, a message confirming non-existence is returned.

    Raises:
        IOError: If the document cannot be found (and not handled by consider_existence)
                 or a connectivity issue occurs.
        ConnectionError: If there's a simulated connection error during retrieval.
    """
    if consider_existence:
        if file_url not in KNOWN_DOCUMENT_URLS:
            return f"Document URL '{file_url}' does not exist in the predetermined list."

    try:
        document_content = retrieve_document_content(file_url)

        if document_content is None:
            # This condition means retrieve_document_content indicated the document was not found
            # This will be raised even if consider_existence was True but the URL was in KNOWN_DOCUMENT_URLS
            raise IOError(f"Document not found at URL: {file_url}")
        
        return document_content

    except ConnectionError as e:
        # Specific handling for connectivity issues if retrieve_document_content raises them
        # Re-raising as IOError as per requirement, or could be a more specific custom exception
        raise IOError(f"Connectivity issue while fetching {file_url}: {e}")
    except Exception as e:
        # Catch any other unexpected errors from retrieve_document_content
        # and wrap them in IOError as per the general requirement for "cannot be found"
        # This might be too broad in a real application.
        raise IOError(f"An unexpected error occurred while fetching {file_url}: {e}")

def validate_document(file_url: str, crypt_signature: str) -> bool:
    """
    Validates a document's integrity using a cryptographic signature.

    Args:
        file_url: The URL of the document.
        crypt_signature: The cryptographic signature to verify.

    Returns:
        True if the signature is verified, False if it does not match.

    Raises:
        IOError: If the document cannot be found at the URL, if there's a connectivity
                 issue, or if no signature is stored for the document.
    """
    try:
        # First, check if the document itself is accessible/exists
        document_content = retrieve_document_content(file_url)
        if document_content is None:
            raise IOError(f"Document not found at URL: {file_url}")

        # If document is accessible, check for its signature
        if file_url not in DOCUMENT_SIGNATURES:
            raise IOError(f"No signature available for document: {file_url}")

        expected_signature = DOCUMENT_SIGNATURES[file_url]
        return expected_signature == crypt_signature

    except ConnectionError as e:
        # Handle connectivity issues from retrieve_document_content
        raise IOError(f"Connectivity issue while accessing document {file_url}: {e}")
    # Other IOErrors (like "Document not found" or "No signature available") will propagate up


if __name__ == '__main__':
    # Test cases
    print("--- Testing fetch_document ---")

    # Test 1: Valid document, consider_existence=False
    valid_url = "http://example.com/doc1"
    print(f"\nFetching: {valid_url} (consider_existence=False)")
    try:
        content = fetch_document(valid_url, consider_existence=False)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 2: Document not found (simulated by None), consider_existence=False
    not_found_url = "http://example.com/missing_doc"
    print(f"\nFetching: {not_found_url} (consider_existence=False)")
    try:
        content = fetch_document(not_found_url, consider_existence=False)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 3: Document URL not in mock data (also not found), consider_existence=False
    unknown_url = "http://example.com/unknown_doc"
    print(f"\nFetching: {unknown_url} (consider_existence=False)")
    try:
        content = fetch_document(unknown_url, consider_existence=False)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 4: Simulated connectivity error, consider_existence=False
    error_url = "http://example.com/error_doc"
    print(f"\nFetching: {error_url} (consider_existence=False)")
    try:
        content = fetch_document(error_url, consider_existence=False)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 5: Another valid document, consider_existence=False
    valid_url_2 = "http://example.com/doc2"
    print(f"\nFetching: {valid_url_2} (consider_existence=False)")
    try:
        content = fetch_document(valid_url_2, consider_existence=False)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    print("\n--- Testing with consider_existence=True ---")

    # Test 6: Valid document, URL in known list, consider_existence=True
    print(f"\nFetching: {valid_url} (consider_existence=True)")
    try:
        content = fetch_document(valid_url, consider_existence=True)
        print(f"Content: {content}") # Should be document content
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 7: Document URL NOT in known list, consider_existence=True
    print(f"\nFetching: {unknown_url} (consider_existence=True)")
    try:
        content = fetch_document(unknown_url, consider_existence=True)
        print(f"Content: {content}") # Should be "does not exist" message
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 8: Document URL in known list but content is None (not found), consider_existence=True
    print(f"\nFetching: {not_found_url} (consider_existence=True)")
    try:
        content = fetch_document(not_found_url, consider_existence=True)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}") # Should raise IOError "Document not found"

    # Test 9: Document URL in known list but causes connectivity error, consider_existence=True
    print(f"\nFetching: {error_url} (consider_existence=True)")
    try:
        content = fetch_document(error_url, consider_existence=True)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}") # Should raise IOError "Connectivity issue"

    print("\n--- Testing validate_document ---")

    # Test VD1: Valid document, valid signature
    print(f"\nValidating: {valid_url} with correct signature")
    try:
        is_valid = validate_document(valid_url, "signature_doc1_valid")
        print(f"Validation result: {is_valid}") # Expected: True
    except IOError as e:
        print(f"Error: {e}")

    # Test VD2: Valid document, invalid signature
    print(f"\nValidating: {valid_url} with incorrect signature")
    try:
        is_valid = validate_document(valid_url, "signature_doc1_invalid")
        print(f"Validation result: {is_valid}") # Expected: False
    except IOError as e:
        print(f"Error: {e}")

    # Test VD3: Document not found (simulated by None in RESOURCE_AVAILABILITY)
    # This document also doesn't have a signature in DOCUMENT_SIGNATURES
    # retrieve_document_content will return None, leading to IOError("Document not found...")
    print(f"\nValidating: {not_found_url} (document not found)")
    try:
        is_valid = validate_document(not_found_url, "any_signature")
        print(f"Validation result: {is_valid}")
    except IOError as e:
        print(f"Error: {e}") # Expected: IOError Document not found...

    # Test VD4: Document URL not in RESOURCE_AVAILABILITY (also not found)
    # retrieve_document_content will return None
    print(f"\nValidating: {unknown_url} (document URL unknown)")
    try:
        is_valid = validate_document(unknown_url, "any_signature")
        print(f"Validation result: {is_valid}")
    except IOError as e:
        print(f"Error: {e}") # Expected: IOError Document not found...

    # Test VD5: Simulated connectivity error
    # retrieve_document_content will raise ConnectionError
    print(f"\nValidating: {error_url} (simulated connectivity error)")
    try:
        is_valid = validate_document(error_url, "any_signature")
        print(f"Validation result: {is_valid}")
    except IOError as e:
        print(f"Error: {e}") # Expected: IOError Connectivity issue...

    # Test VD6: Valid document (doc2), but no signature provided in DOCUMENT_SIGNATURES for it
    # For this test, let's assume doc2 *is* retrievable but we *don't* have its signature in DOCUMENT_SIGNATURES
    # To test this, we need a doc that is in RESOURCE_AVAILABILITY but not DOCUMENT_SIGNATURES
    # Let's add a temporary doc for this or modify an existing one.
    # Current DOCUMENT_SIGNATURES only has doc1 and doc2.
    # Let's test with doc2, assuming it's retrievable.
    # If doc2 is in RESOURCE_AVAILABILITY and DOCUMENT_SIGNATURES, this test is like VD1/VD2.
    # The current setup: doc2 is in RESOURCE_AVAILABILITY and DOCUMENT_SIGNATURES.
    # To test "No signature available", we need a URL that is in RESOURCE_AVAILABILITY
    # but NOT in DOCUMENT_SIGNATURES.
    # Let's imagine a "http://example.com/doc_no_sig" which is retrievable but has no signature.
    RESOURCE_AVAILABILITY["http://example.com/doc_no_sig"] = "Content of doc with no signature"
    doc_no_sig_url = "http://example.com/doc_no_sig"
    print(f"\nValidating: {doc_no_sig_url} (document exists, but no signature stored)")
    try:
        is_valid = validate_document(doc_no_sig_url, "any_signature")
        print(f"Validation result: {is_valid}")
    except IOError as e:
        print(f"Error: {e}") # Expected: IOError No signature available...
    del RESOURCE_AVAILABILITY["http://example.com/doc_no_sig"] # Clean up test data

    # Test VD7: Valid document (doc2), valid signature
    print(f"\nValidating: {valid_url_2} with correct signature")
    try:
        is_valid = validate_document(valid_url_2, "signature_doc2_valid")
        print(f"Validation result: {is_valid}") # Expected: True
    except IOError as e:
        print(f"Error: {e}")
