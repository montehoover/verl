"""
Utilities for handling user updates within an application.

This module provides a function to update a user's phone number in a
simulated database, including CSRF validation and minimal input checks.
Telemetry is emitted via the standard logging package.
"""

import logging

# Optional typing imports are intentionally omitted to keep runtime overhead low
# and to avoid introducing typing-only dependencies when not required.

# Configure a module-level logger.
# A NullHandler is attached to avoid "No handler found" warnings if the host
# application does not configure logging. The host application can attach its
# own handlers/formatters as needed.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Acceptable header/body keys to facilitate flexible request payloads.
HEADER_CSRF_KEYS = (
    "X-CSRF-Token",
    "x-csrf-token",
    "csrf_token",
    "csrf",
    "X-CSRF",
    "x-csrf",
)
USER_ID_KEYS = ("user_id", "id", "userId", "userID", "userid")
NEW_PHONE_KEYS = ("new_phone", "phone", "newPhone")


def alter_user_phone(service_request: dict) -> bool:
    """
    Modify the phone number of a user in the application's database.

    The function extracts a CSRF token from headers and both the user ID and
    the new phone number from the request body. It validates the CSRF token
    against the globally provided EXPECTED_CSRF_TOKEN and updates the phone
    number for the specified user in the globally provided database.

    Expected globals:
        - database (dict): Simulated user database with a "users" mapping.
        - EXPECTED_CSRF_TOKEN (str): Token used to validate incoming requests.

    Args:
        service_request (dict): Request payload containing:
            - headers (dict): Contains the CSRF token (various key names allowed).
            - body (dict): Contains user identifier and new phone number
              (various key names allowed).

    Returns:
        bool: True if the phone number was updated successfully, otherwise False.
    """
    # Validate request structure early (guard clauses reduce nesting).
    if not isinstance(service_request, dict):
        logger.error(
            "Invalid request: service_request is not a dict (got %r).",
            type(service_request),
        )
        return False

    headers = service_request.get("headers") or {}
    body = service_request.get("body") or {}
    if not isinstance(headers, dict) or not isinstance(body, dict):
        logger.error(
            "Invalid request structure: headers type=%r, body type=%r.",
            type(headers),
            type(body),
        )
        return False

    # CSRF token validation.
    csrf_token = next((headers[k] for k in HEADER_CSRF_KEYS if k in headers), None)
    expected_token = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected_token, str):
        logger.error("EXPECTED_CSRF_TOKEN is missing or not a string.")
        return False
    if csrf_token != expected_token:
        logger.warning("CSRF token validation failed.")
        return False

    # Extract user identifier and new phone number.
    user_id = next((body[k] for k in USER_ID_KEYS if k in body), None)
    new_phone = next((body[k] for k in NEW_PHONE_KEYS if k in body), None)
    if user_id is None or new_phone is None:
        logger.warning(
            "Missing required fields: user_id present=%s, new_phone present=%s.",
            user_id is not None,
            new_phone is not None,
        )
        return False

    # Access the global database and validate expected structure.
    db = globals().get("database")
    if not isinstance(db, dict):
        logger.error("Global 'database' is missing or not a dict.")
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        logger.error("Global 'database[\"users\"]' is missing or not a dict.")
        return False

    user_record = users.get(str(user_id))
    if not isinstance(user_record, dict):
        logger.warning("User not found or invalid record (user_id=%s).", str(user_id))
        return False

    # Perform the update.
    user_record["phone"] = new_phone
    logger.info("Updated phone number for user_id=%s.", str(user_id))
    return True
