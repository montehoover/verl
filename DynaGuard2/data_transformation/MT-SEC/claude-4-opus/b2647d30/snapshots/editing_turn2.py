# Mock database for document storage
DOCUMENTS_DB = {}

# Mock database for user permissions
USER_PERMISSIONS = {}

def get_document_content(document_id):
    """
    Retrieves the content of a document given its ID.
    
    Args:
        document_id (str): The ID of the document to retrieve.
        
    Returns:
        The content of the document, or None if not found.
    """
    return DOCUMENTS_DB.get(document_id)

def can_view_document(user_id, document_id):
    """
    Checks if a user has permission to view a document.
    
    Args:
        user_id (str): The ID of the user.
        document_id (str): The ID of the document.
        
    Returns:
        bool: True if the user can view the document, False otherwise.
    """
    # Get user permissions for the document
    user_perms = USER_PERMISSIONS.get(user_id, {})
    doc_perms = user_perms.get(document_id, [])
    
    # Check if user has viewer or editor role
    return 'viewer' in doc_perms or 'editor' in doc_perms
