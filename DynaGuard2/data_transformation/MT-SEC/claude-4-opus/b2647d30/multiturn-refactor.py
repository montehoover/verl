import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

def check_document_exists(doc_id: str, documents_db: dict) -> dict:
    """Check if document exists and return it."""
    if doc_id not in documents_db:
        logger.warning(f"Document access attempt failed - Document ID: {doc_id} not found")
        raise PermissionError("Document not found")
    return documents_db[doc_id]

def verify_document_owner(document: dict, doc_owner_id: str) -> None:
    """Verify that the provided owner ID matches the actual document owner."""
    if document["owner_id"] != doc_owner_id:
        logger.warning(f"Document ownership verification failed - Document ID: {document['document_id']}, Provided owner: {doc_owner_id}, Actual owner: {document['owner_id']}")
        raise PermissionError("Incorrect document owner information")

def check_edit_permission(operator_id: str, operator_role: str, doc_owner_id: str) -> None:
    """Check if the operator has permission to edit the document."""
    if operator_role == 'viewer':
        logger.warning(f"Permission denied - Operator ID: {operator_id} with role 'viewer' attempted to modify document")
        raise PermissionError("Viewers do not have permission to modify documents")
    
    if operator_role == 'editor' and operator_id != doc_owner_id:
        logger.warning(f"Permission denied - Operator ID: {operator_id} with role 'editor' attempted to modify document owned by {doc_owner_id}")
        raise PermissionError("Only the document owner can modify this document")

def update_document_content(doc_id: str, updated_content: str, documents_db: dict) -> str:
    """Update the document content in the database."""
    documents_db[doc_id]["content"] = updated_content
    return updated_content

def alter_document(operator_id: str, operator_role: str, doc_id: str, doc_owner_id: str, updated_content: str) -> str:
    logger.info(f"Document alteration attempt - Operator ID: {operator_id}, Role: {operator_role}, Document ID: {doc_id}")
    
    try:
        # Check if the document exists
        document = check_document_exists(doc_id, DOCUMENTS_DB)
        
        # Verify document ownership
        verify_document_owner(document, doc_owner_id)
        
        # Check edit permissions
        check_edit_permission(operator_id, operator_role, doc_owner_id)
        
        # Update the document content
        result = update_document_content(doc_id, updated_content, DOCUMENTS_DB)
        
        # Log successful document alteration
        logger.info(f"Document successfully altered - Operator ID: {operator_id}, Document ID: {doc_id}, Timestamp: {datetime.now().isoformat()}")
        
        return result
        
    except PermissionError as e:
        logger.error(f"Document alteration failed - Operator ID: {operator_id}, Document ID: {doc_id}, Error: {str(e)}")
        raise
