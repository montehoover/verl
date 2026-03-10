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
