import logging
from datetime import datetime

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def _check_permissions(operator_id: str, operator_role: str, doc_owner_id: str, actual_owner_id_from_db: str) -> bool:
    """
    Checks if the operator has permission to alter the document.

    Args:
        operator_id: The ID of the user trying to modify the document.
        operator_role: The role of the user.
        doc_owner_id: The ID of the document's owner (as asserted by the caller).
        actual_owner_id_from_db: The actual owner ID of the document from the database.

    Returns:
        True if the operator has permission, False otherwise.
    """
    is_editor = (operator_role == 'editor')
    is_operator_the_owner = (operator_id == actual_owner_id_from_db)
    is_provided_owner_correct = (doc_owner_id == actual_owner_id_from_db)
    
    return is_editor and is_operator_the_owner and is_provided_owner_correct


def _update_document_content_in_db(document_record: dict, updated_content: str) -> str:
    """
    Updates the content of the document record and returns the new content.

    Args:
        document_record: The document record dictionary from DOCUMENTS_DB.
        updated_content: The new content for the document.

    Returns:
        The updated content of the document.
    """
    document_record['content'] = updated_content
    return updated_content


def alter_document(operator_id: str, operator_role: str, doc_id: str, doc_owner_id: str, updated_content: str) -> str:
    """
    Changes the content of an existing document if the operator is authorized.

    Args:
        operator_id: The ID of the user trying to modify the document.
        operator_role: The role of the user (e.g., 'viewer', 'editor').
        doc_id: The unique ID of the document to be changed.
        doc_owner_id: The ID of the document's owner (as asserted by the caller).
        updated_content: The new content for the document.

    Returns:
        The updated content of the document.

    Raises:
        PermissionError: If the operator is not authorized to modify the document,
                         or if the document is not found, or if owner information is inconsistent.
    """
    if doc_id not in DOCUMENTS_DB:
        raise PermissionError("Insufficient permissions")

    document_record = DOCUMENTS_DB[doc_id]
    actual_owner_id_from_db = document_record['owner_id']

    if _check_permissions(operator_id, operator_role, doc_owner_id, actual_owner_id_from_db):
        updated_doc = _update_document_content_in_db(document_record, updated_content)
        # Log the successful alteration
        timestamp = datetime.now().isoformat()
        logging.info(
            f"Document altered: operator_id='{operator_id}', doc_id='{doc_id}', timestamp='{timestamp}'"
        )
        return updated_doc
    else:
        raise PermissionError("Insufficient permissions")
