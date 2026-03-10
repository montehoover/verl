from typing import Any


__all__ = ["get_document_content", "can_view_document", "alter_document"]


def get_document_content(document_id: str) -> str:
    """
    Retrieve the content of a document by its ID from the global DOCUMENTS_DB.

    Args:
        document_id (str): The ID of the document.

    Returns:
        str: The content of the document.

    Raises:
        TypeError: If document_id is not a string.
        KeyError: If the document_id does not exist in DOCUMENTS_DB.
        NameError: If DOCUMENTS_DB is not defined in the environment.
    """
    if not isinstance(document_id, str):
        raise TypeError("document_id must be a string")

    # Access via globals() to avoid referencing an undefined name at import time
    try:
        db = globals()["DOCUMENTS_DB"]
    except KeyError:
        raise NameError("DOCUMENTS_DB is not defined") from None

    return db[document_id]


def can_view_document(user_id: str, document_id: str) -> bool:
    """
    Determine whether a user is allowed to view a document based on role assignments.

    Expected global structure:
        DOCUMENT_ROLES_DB: Dict[str, Dict[str, str | Iterable[str]]]
            Mapping of document_id -> { user_id -> role or roles }
            Example role values include: 'viewer', 'editor', 'owner', 'admin'.

    Args:
        user_id (str): The ID of the user.
        document_id (str): The ID of the document.

    Returns:
        bool: True if the user has a role that permits viewing, otherwise False.

    Raises:
        TypeError: If user_id or document_id is not a string.
        NameError: If DOCUMENT_ROLES_DB is not defined in the environment.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not isinstance(document_id, str):
        raise TypeError("document_id must be a string")

    try:
        roles_db = globals()["DOCUMENT_ROLES_DB"]
    except KeyError:
        raise NameError("DOCUMENT_ROLES_DB is not defined") from None

    doc_roles = roles_db.get(document_id)
    if not isinstance(doc_roles, dict):
        return False

    role = doc_roles.get(user_id)
    if role is None:
        return False

    allowed_roles = {"viewer", "editor", "owner", "admin"}

    if isinstance(role, str):
        return role in allowed_roles

    # If multiple roles are provided as an iterable, allow if any role is permitted
    try:
        return any(r in allowed_roles for r in role)
    except TypeError:
        # role is neither a string nor an iterable of roles
        return False


def alter_document(
    operator_id: str,
    operator_role: str,
    doc_id: str,
    doc_owner_id: str,
    updated_content: str,
) -> str:
    """
    Alter the content of an existing document if the operator is authorized.

    Authorization rules:
      - The operator can modify the document if they are the owner of the document
        OR their role is 'editor'.
      - A 'viewer' cannot modify unless they are the owner.

    Args:
        operator_id (str): ID of the user attempting the change.
        operator_role (str): Role of the user ('viewer' or 'editor').
        doc_id (str): Unique ID of the document to be changed.
        doc_owner_id (str): ID of the document's owner (informational).
        updated_content (str): New content to replace the current version.

    Returns:
        str: The updated content.

    Raises:
        TypeError: If any argument is not a string.
        NameError: If DOCUMENTS_DB is not defined.
        KeyError: If the document ID is not found in DOCUMENTS_DB.
        PermissionError: If the operator is not authorized to modify the document.
    """
    # Type checks
    if not isinstance(operator_id, str):
        raise TypeError("operator_id must be a string")
    if not isinstance(operator_role, str):
        raise TypeError("operator_role must be a string")
    if not isinstance(doc_id, str):
        raise TypeError("doc_id must be a string")
    if not isinstance(doc_owner_id, str):
        raise TypeError("doc_owner_id must be a string")
    if not isinstance(updated_content, str):
        raise TypeError("updated_content must be a string")

    # Access documents database
    try:
        db = globals()["DOCUMENTS_DB"]
    except KeyError:
        raise NameError("DOCUMENTS_DB is not defined") from None

    # Ensure document exists
    doc = db[doc_id]

    # Determine actual owner from the DB (authoritative source)
    actual_owner_id = doc.get("owner_id")

    # Normalize role for comparison
    role_normalized = operator_role.lower()

    # Authorization: owner or editor can modify
    is_owner = operator_id == actual_owner_id
    is_editor = role_normalized == "editor"

    if not (is_owner or is_editor):
        raise PermissionError("Operator is not authorized to alter this document")

    # Perform the update
    doc["content"] = updated_content
    return updated_content
