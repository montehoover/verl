"""
Refactored utilities for updating a user's email address using a request-like payload.

This module applies the Compose Method pattern:
- The public orchestrator (alter_user_email) coordinates the workflow.
- Pure helper functions (_validate_csrf, _extract_email, _apply_email_update) each handle a single concern.
- A lightweight parameter object (RequestParams) centralizes and normalizes payload data extraction.

Side effects (writing to the storage) are confined to the orchestrator.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


@dataclass(frozen=True)
class RequestParams:
    """
    Immutable parameter object encapsulating normalized request data.

    Attributes:
        headers (dict): Request headers mapping.
        body (dict): Request body mapping.
        user_id (str): Normalized user identifier (always a string).
    """
    headers: dict
    body: dict
    user_id: str


def _validate_csrf(headers: dict, expected_token: str) -> bool:
    """
    Pure function: validates the CSRF token from headers.

    Args:
        headers (dict): Request headers.
        expected_token (str): The expected CSRF token to compare against.

    Returns:
        bool: True when the provided token matches the expected value, otherwise False.
    """
    if not isinstance(headers, dict):
        return False

    # Support common header naming styles for CSRF tokens
    csrf_token = headers.get("X-CSRF-Token") or headers.get("csrf_token")
    return csrf_token == expected_token


def _extract_email(body: dict) -> Optional[str]:
    """
    Pure function: extracts and validates the new email from the request body.

    The function performs minimal validation (presence of '@', no spaces, non-empty)
    and returns a normalized email string.

    Args:
        body (dict): Request body.

    Returns:
        Optional[str]: Normalized email if valid; otherwise None.
    """
    if not isinstance(body, dict):
        return None

    # Accept common keys for new email
    email = body.get("email") or body.get("new_email")
    if not isinstance(email, str):
        return None

    email = email.strip()
    # Minimal constraints: non-empty, contains '@', no spaces
    if not email or "@" not in email or " " in email:
        return None

    return email


def _apply_email_update(users: dict, user_id: str, new_email: str) -> Tuple[dict, bool]:
    """
    Pure function: returns a new users mapping with the email updated for the given user_id.

    This function avoids any side effects by:
    - Not mutating the input mapping.
    - Returning a new mapping instance when successful.

    Args:
        users (dict): A mapping of user_id -> user_record (dict).
        user_id (str): The user identifier to update.
        new_email (str): The new email to set.

    Returns:
        Tuple[dict, bool]: (updated_users_mapping, success_flag)
            - updated_users_mapping: A shallow copy of users with the updated record when successful;
              otherwise the original users mapping.
            - success_flag: True when the user exists and was updated; False otherwise.
    """
    if not isinstance(users, dict) or user_id not in users:
        return users, False

    # Shallow copy of the users mapping
    updated_users = users.copy()

    # Copy the user record to avoid mutating nested structures from the original mapping
    user_record = dict(updated_users[user_id])
    user_record["email"] = new_email

    # Reassign the updated record back into the copied users mapping
    updated_users[user_id] = user_record
    return updated_users, True


def _parse_payload(payload: dict) -> Optional[RequestParams]:
    """
    Pure function: parses and normalizes the incoming payload into a parameter object.

    Responsibilities:
    - Ensure payload is a dict.
    - Extract headers/body with safe defaults.
    - Extract and normalize user_id as a string.

    Args:
        payload (dict): Raw request-like payload containing 'headers' and 'body'.

    Returns:
        Optional[RequestParams]: A RequestParams instance if parsing and normalization succeed;
                                 otherwise None.
    """
    if not isinstance(payload, dict):
        return None

    # Defensive extraction with safe defaults
    headers = payload.get("headers") or {}
    body = payload.get("body") or {}

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return None

    # Extract user identifier and normalize to string (storage uses str keys)
    user_id = body.get("user_id")
    if user_id is None:
        return None
    user_id = str(user_id)

    # Create an immutable parameter object to pass through the pipeline
    return RequestParams(headers=dict(headers), body=dict(body), user_id=user_id)


def alter_user_email(payload: dict) -> bool:
    """
    Modify a user's email address using data from a request-like payload.

    Orchestration steps:
    1) Parse and normalize the payload into a parameter object.
    2) Validate CSRF token from headers.
    3) Extract and validate the new email from the body.
    4) Apply the email update in a pure function.
    5) Commit the mutation to the storage if successful.

    Args:
        payload (dict): The request containing headers and body data with the user ID and new email address.

    Returns:
        bool: True if the update is successful, False otherwise.
    """
    try:
        # Step 1: Parse and normalize payload
        params = _parse_payload(payload)
        if params is None:
            return False

        # Step 2: CSRF validation
        if not _validate_csrf(params.headers, EXPECTED_CSRF_TOKEN):
            return False

        # Step 3: Extract and validate new email
        new_email = _extract_email(params.body)
        if new_email is None:
            return False

        # Step 4: Obtain current users mapping and apply update using a pure function
        users = record_storage.get("users", {})
        if not isinstance(users, dict):
            return False

        updated_users, success = _apply_email_update(users, params.user_id, new_email)
        if not success:
            return False

        # Step 5: Commit the updated users mapping back to storage (side effect isolated here)
        record_storage["users"] = updated_users
        return True
    except Exception:
        # Fail-closed: return False on unexpected errors
        return False
