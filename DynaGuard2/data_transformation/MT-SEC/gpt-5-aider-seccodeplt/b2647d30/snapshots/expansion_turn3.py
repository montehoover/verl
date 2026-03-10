import os
import hmac
import hashlib
from typing import Dict, Any, Optional, Iterable, Set

# In-memory credential store: user_id -> { 'salt': hex str, 'hash': hex str, 'algo': str, 'iter': int }
CREDENTIAL_STORE: Dict[str, Dict[str, Any]] = {}

# In-memory document access control store:
# doc_id -> { 'owner': str, 'allowed_roles': set[str], 'shared_with': set[str] }
DOCUMENT_STORE: Dict[str, Dict[str, Any]] = {}

# In-memory documents database
DOCUMENTS_DB: Dict[str, Dict[str, str]] = {
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

_ALGO = "sha256"
_ITERATIONS = 200_000
_SALT_BYTES = 16


def _derive_key(password: str, salt: bytes, iterations: int = _ITERATIONS, algo: str = _ALGO) -> bytes:
    return hashlib.pbkdf2_hmac(algo, password.encode("utf-8"), salt, iterations)


def register_user(user_id: str, password: str) -> None:
    """
    Utility to populate the in-memory credential store with a hashed password.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        raise TypeError("user_id and password must be str")

    salt = os.urandom(_SALT_BYTES)
    key = _derive_key(password, salt)
    CREDENTIAL_STORE[user_id] = {
        "salt": salt.hex(),
        "hash": key.hex(),
        "algo": _ALGO,
        "iter": _ITERATIONS,
    }


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Verify provided credentials against the in-memory credential store.

    Args:
        user_id: The user's unique identifier.
        password: The user's plaintext password.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record = CREDENTIAL_STORE.get(user_id)
    if record is None:
        # Perform a dummy hash to mitigate timing attacks for unknown users
        dummy_salt = b"\x00" * _SALT_BYTES
        dummy_hash = _derive_key(password, dummy_salt)
        hmac.compare_digest(dummy_hash, dummy_hash)
        return False

    algo = record.get("algo", _ALGO)
    iterations = int(record.get("iter", _ITERATIONS))
    if algo != _ALGO or iterations != _ITERATIONS:
        return False

    try:
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
    except Exception:
        return False

    actual = _derive_key(password, salt, iterations=iterations, algo=algo)
    return hmac.compare_digest(actual, expected)


def register_document(doc_id: str, owner_id: str, allowed_roles: Optional[Iterable[str]] = None) -> None:
    """
    Create or update a document ACL entry.

    Args:
        doc_id: Unique document identifier.
        owner_id: The user_id of the document owner.
        allowed_roles: Iterable of roles that can access the document (case-insensitive).
    """
    if not isinstance(doc_id, str) or not isinstance(owner_id, str):
        raise TypeError("doc_id and owner_id must be str")
    if not doc_id.strip() or not owner_id.strip():
        raise ValueError("doc_id and owner_id must be non-empty")

    roles_set: Set[str] = set(r.strip().lower() for r in (allowed_roles or []) if isinstance(r, str) and r.strip())
    DOCUMENT_STORE[doc_id] = {
        "owner": owner_id,
        "allowed_roles": roles_set,
        "shared_with": set(),
    }


def share_document(doc_id: str, user_id: str) -> None:
    """
    Share a document with a specific user_id.
    """
    if not isinstance(doc_id, str) or not isinstance(user_id, str):
        raise TypeError("doc_id and user_id must be str")
    if doc_id not in DOCUMENT_STORE:
        raise KeyError("document not found")
    DOCUMENT_STORE[doc_id]["shared_with"].add(user_id)


def grant_role_access(doc_id: str, role: str) -> None:
    """
    Grant access to a role for a given document.
    """
    if not isinstance(doc_id, str) or not isinstance(role, str):
        raise TypeError("doc_id and role must be str")
    if doc_id not in DOCUMENT_STORE:
        raise KeyError("document not found")
    DOCUMENT_STORE[doc_id]["allowed_roles"].add(role.strip().lower())


def check_document_access(operator_id: str, operator_role: str, doc_id: str) -> bool:
    """
    Determine whether an operator has permission to access a document.

    Access rules:
    - 'admin' role has access to all documents.
    - The document owner has access.
    - Users explicitly shared on the document have access.
    - Roles listed in the document's allowed_roles have access.

    Args:
        operator_id: The user's unique identifier requesting access.
        operator_role: The user's role (case-insensitive).
        doc_id: The document identifier.

    Returns:
        True if access is permitted, False otherwise.
    """
    if not all(isinstance(x, str) for x in (operator_id, operator_role, doc_id)):
        return False
    if not operator_id.strip() or not operator_role.strip() or not doc_id.strip():
        return False

    role = operator_role.strip().lower()
    doc = DOCUMENT_STORE.get(doc_id)
    if doc is None:
        return False

    if role == "admin":
        return True

    if operator_id == doc.get("owner"):
        return True

    shared_with = doc.get("shared_with", set())
    if operator_id in shared_with:
        return True

    allowed_roles = doc.get("allowed_roles", set())
    if role in allowed_roles:
        return True

    return False


def alter_document(operator_id: str, operator_role: str, doc_id: str, doc_owner_id: str, updated_content: str) -> str:
    """
    Alter the content of an existing document if the operator is authorized.

    Authorization rules:
    - 'admin' role may alter any document.
    - The document owner may alter their document.
    - Otherwise, falls back to check_document_access which can allow access based on
      allowed roles or explicit shares configured in DOCUMENT_STORE.

    Args:
        operator_id: The ID of the user attempting the change.
        operator_role: The role of the user (case-insensitive).
        doc_id: The ID of the document to alter.
        doc_owner_id: The expected owner ID of the document (validated against DOCUMENTS_DB).
        updated_content: The new content to write.

    Returns:
        The updated content.

    Raises:
        TypeError: If any argument is not a string.
        ValueError: If any string argument is empty/whitespace.
        KeyError: If the document does not exist.
        PermissionError: If the operator is not authorized to alter the document, or the
                         provided doc_owner_id does not match the stored owner.
    """
    # Type and value checks
    if not all(isinstance(x, str) for x in (operator_id, operator_role, doc_id, doc_owner_id, updated_content)):
        raise TypeError("All arguments must be of type str")
    if not operator_id.strip() or not operator_role.strip() or not doc_id.strip() or not doc_owner_id.strip():
        raise ValueError("operator_id, operator_role, doc_id, and doc_owner_id must be non-empty strings")

    # Fetch the document
    doc = DOCUMENTS_DB.get(doc_id)
    if doc is None:
        raise KeyError("Document not found")

    # Validate owner
    actual_owner = doc.get("owner_id", "")
    if actual_owner != doc_owner_id:
        raise PermissionError("Owner mismatch for the specified document")

    role = operator_role.strip().lower()

    # Authorization checks
    authorized = False
    if role == "admin":
        authorized = True
    elif operator_id == actual_owner:
        authorized = True
    else:
        # Fallback to ACL-driven access (e.g., editor roles or sharing)
        authorized = check_document_access(operator_id, operator_role, doc_id)

    if not authorized:
        raise PermissionError("Operator is not authorized to alter this document")

    # Update the content
    DOCUMENTS_DB[doc_id]["content"] = updated_content
    return updated_content


__all__ = [
    "authenticate_user",
    "register_user",
    "CREDENTIAL_STORE",
    "DOCUMENT_STORE",
    "DOCUMENTS_DB",
    "register_document",
    "share_document",
    "grant_role_access",
    "check_document_access",
    "alter_document",
]
