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

def fetch_document(file_url: str) -> str:
    """
    Retrieves data from a remote server.

    Args:
        file_url: The URL of the document to retrieve.

    Returns:
        The document content if the URL is valid and the document is found.

    Raises:
        IOError: If the document cannot be found or a connectivity issue occurs.
        ConnectionError: If there's a simulated connection error during retrieval.
    """
    try:
        # Check resource availability first (optional, depending on how retrieve_document_content works)
        # For this example, we'll assume retrieve_document_content handles "not found" by returning None
        # and other issues by raising exceptions.

        document_content = retrieve_document_content(file_url)

        if document_content is None:
            # This condition means retrieve_document_content indicated the document was not found
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

    # Test 1: Valid document
    valid_url = "http://example.com/doc1"
    print(f"\nFetching: {valid_url}")
    try:
        content = fetch_document(valid_url)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 2: Document not found (simulated by None in RESOURCE_AVAILABILITY)
    not_found_url = "http://example.com/missing_doc"
    print(f"\nFetching: {not_found_url}")
    try:
        content = fetch_document(not_found_url)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 3: Document URL not in our mock data (also not found)
    unknown_url = "http://example.com/unknown_doc"
    print(f"\nFetching: {unknown_url}")
    try:
        content = fetch_document(unknown_url)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 4: Simulated connectivity error
    error_url = "http://example.com/error_doc"
    print(f"\nFetching: {error_url}")
    try:
        content = fetch_document(error_url)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")

    # Test 5: Another valid document
    valid_url_2 = "http://example.com/doc2"
    print(f"\nFetching: {valid_url_2}")
    try:
        content = fetch_document(valid_url_2)
        print(f"Content: {content}")
    except (IOError, ConnectionError) as e:
        print(f"Error: {e}")
