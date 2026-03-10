# Mock database for document storage
DOCUMENTS_DB = {}

def get_document_content(document_id):
    """
    Retrieves the content of a document given its ID.
    
    Args:
        document_id (str): The ID of the document to retrieve.
        
    Returns:
        The content of the document, or None if not found.
    """
    return DOCUMENTS_DB.get(document_id)
