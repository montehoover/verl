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

def alter_document(operator_id, operator_role, doc_id, doc_owner_id, updated_content):
    """
    Allows a user to change the content of an existing document.
    
    Args:
        operator_id (str): The ID of the user trying to modify the document.
        operator_role (str): The role of the user (either 'viewer' or 'editor').
        doc_id (str): The unique ID of the document to be changed.
        doc_owner_id (str): The ID of the document's owner.
        updated_content (str): The new content that will replace the current version.
        
    Returns:
        str: The updated content if the user is authorized.
        
    Raises:
        PermissionError: If the user is not authorized to modify the document.
    """
    # Check if the operator is the owner or has editor role
    if operator_id == doc_owner_id or operator_role == 'editor':
        # Update the document content
        if doc_id in DOCUMENTS_DB:
            DOCUMENTS_DB[doc_id]['content'] = updated_content
            return updated_content
        else:
            raise PermissionError("Document not found")
    else:
        raise PermissionError("You do not have permission to modify this document")
