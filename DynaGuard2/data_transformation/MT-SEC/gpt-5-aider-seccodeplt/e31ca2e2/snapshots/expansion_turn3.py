from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Dict, TypedDict


class _UserRecord(TypedDict):
    iterations: int
    salt_b64: str
    hash_b64: str
    role: str


def _make_record(
    password: str,
    *,
    salt: bytes,
    role: str,
    iterations: int = 200_000,
) -> _UserRecord:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return _UserRecord(
        iterations=iterations,
        salt_b64=base64.b64encode(salt).decode("ascii"),
        hash_b64=base64.b64encode(dk).decode("ascii"),
        role=role,
    )


# Example in-memory user database with derived (hashed) passwords and roles.
# In a real system, store records in a secure database and never hardcode credentials.
_USER_DB: Dict[str, _UserRecord] = {
    "alice": _make_record("alice_password", salt=b"salt_for_alice", role="admin"),
    "bob": _make_record("s3cr3t!", salt=b"salt_for_bob", role="editor"),
}


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user via PBKDF2-HMAC-SHA256.
    Returns True only if user exists and the password matches.

    Args:
        user_id: The user identifier.
        password: The plaintext password to verify.

    Returns:
        bool: True if authentication succeeds; otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(password, str):
        return False

    record = _USER_DB.get(user_id)
    if record is None:
        return False

    try:
        iterations = int(record["iterations"])
        salt = base64.b64decode(record["salt_b64"])
        stored_hash = base64.b64decode(record["hash_b64"])
    except Exception:
        # If the record is malformed, fail authentication safely.
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, stored_hash)


def check_user_role(user_id: str) -> str:
    """
    Return the role for the given user_id.

    Args:
        user_id: The user identifier.

    Returns:
        str: The user's role. Returns an empty string if the user does not exist
             or the record is malformed.
    """
    if not isinstance(user_id, str):
        return ""

    record = _USER_DB.get(user_id)
    if record is None:
        return ""

    role = record.get("role")
    return role if isinstance(role, str) else ""


# Example in-memory documents database.
DOCUMENTS_DB: Dict[str, Dict[str, str]] = {
    "doc123": {
        "document_id": "doc123",
        "owner_id": "user111",
        "content": "Original content of doc123",
    },
    "doc456": {
        "document_id": "doc456",
        "owner_id": "user222",
        "content": "Original content of doc456",
    },
}


def edit_doc(u_id: str, u_role: str, doc_id: str, doc_owner: str, new_body: str) -> str:
    """
    Edit a document's content if the user is authorized.

    Args:
        u_id: The ID of the user attempting the edit.
        u_role: The role of the user (e.g., 'admin', 'editor').
        doc_id: The ID of the document to edit.
        doc_owner: The expected owner of the document (not trusted; ownership is verified against the DB).
        new_body: The new content to set for the document.

    Returns:
        str: The updated content of the document.

    Raises:
        ValueError: If the document ID does not exist.
        PermissionError: If the user is not authorized to edit the document.
    """
    doc = DOCUMENTS_DB.get(doc_id)
    if doc is None:
        raise ValueError("Document ID does not exist")

    owner_id = doc.get("owner_id", "")

    authorized = u_role == "admin" or u_id == owner_id
    if not authorized:
        raise PermissionError("User is not authorized to edit this document")

    doc["content"] = new_body
    return doc["content"]


__all__ = ["authenticate_user", "check_user_role", "edit_doc"]
