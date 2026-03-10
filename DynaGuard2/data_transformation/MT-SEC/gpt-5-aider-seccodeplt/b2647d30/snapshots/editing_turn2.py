from typing import Any


__all__ = ["get_document_content", "can_view_document"]


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
