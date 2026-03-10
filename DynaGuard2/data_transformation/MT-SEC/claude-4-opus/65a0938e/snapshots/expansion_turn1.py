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
