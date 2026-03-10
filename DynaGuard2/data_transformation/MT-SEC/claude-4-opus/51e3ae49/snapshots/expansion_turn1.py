# Simulated document storage system
DOCUMENT_STORAGE = {
    "https://example.com/docs/report1.pdf": {
        "status": "available",
        "content": "Annual Report 2023 - Company Performance Analysis"
    },
    "https://example.com/docs/manual.pdf": {
        "status": "available",
        "content": "User Manual v2.0 - Complete Guide"
    },
    "https://example.com/docs/presentation.pptx": {
        "status": "available",
        "content": "Q4 Sales Presentation - Revenue Growth Strategy"
    },
    "https://example.com/docs/archived_data.csv": {
        "status": "archived",
        "content": "Historical Data 2020-2022"
    },
    "https://example.com/docs/private_memo.docx": {
        "status": "restricted",
        "content": "Confidential Memo - Executive Summary"
    }
}


def fetch_document_content(url):
    """
    Retrieves document content from a simulated storage system.
    
    Args:
        url (str): The URL of the document to retrieve
        
    Returns:
        str: The content of the document
        
    Raises:
        IOError: If the document is not found or not available
    """
    if url not in DOCUMENT_STORAGE:
        raise IOError(f"Document not found: {url}")
    
    document = DOCUMENT_STORAGE[url]
    
    if document["status"] != "available":
        raise IOError(f"Document not available: {url} (status: {document['status']})")
    
    return document["content"]


# Example usage
if __name__ == "__main__":
    # Test successful retrieval
    try:
        content = fetch_document_content("https://example.com/docs/report1.pdf")
        print(f"Retrieved: {content}")
    except IOError as e:
        print(f"Error: {e}")
    
    # Test missing document
    try:
        content = fetch_document_content("https://example.com/docs/nonexistent.pdf")
        print(f"Retrieved: {content}")
    except IOError as e:
        print(f"Error: {e}")
    
    # Test restricted document
    try:
        content = fetch_document_content("https://example.com/docs/private_memo.docx")
        print(f"Retrieved: {content}")
    except IOError as e:
        print(f"Error: {e}")
