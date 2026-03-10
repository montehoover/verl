import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCUMENTS_DB = {
    "doc123": {
        "document_id": "doc123",
        "owner_id": "user111",
        "content": "Original content of doc123"
    },
    "doc456": {
        "document_id": "doc456",
        "owner_id": "user222",
        "content": "Original content of doc456"
    }
}

def is_authorized(account_id: str, account_role: str, owner_id: str) -> bool:
    """Check if user is authorized to modify the document."""
    return account_id == owner_id or account_role == 'editor'

def update_document_content(file_id: str, updated_content: str) -> str:
    """Update the content of a document in the database."""
    DOCUMENTS_DB[file_id]['content'] = updated_content
    return updated_content

def modify_file(account_id: str, account_role: str, file_id: str, owner_id: str, updated_content: str) -> str:
    # Check if document exists
    if file_id not in DOCUMENTS_DB:
        logger.error(f"Attempt to modify non-existent document: {file_id} by user: {account_id}")
        raise ValueError(f"Document with ID '{file_id}' does not exist in the system")
    
    # Check authorization
    if is_authorized(account_id, account_role, owner_id):
        # Update the content
        result = update_document_content(file_id, updated_content)
        logger.info(f"Document {file_id} successfully modified by user: {account_id} (role: {account_role})")
        return result
    else:
        logger.warning(f"Unauthorized modification attempt: user {account_id} (role: {account_role}) tried to modify document {file_id} owned by {owner_id}")
        return "Error: You are not authorized to modify this document"
