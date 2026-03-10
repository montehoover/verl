from typing import Optional
from dataclasses import dataclass

# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

@dataclass
class UserEmailPayload:
    """
    Represents the structured data required for updating a user's email.
    
    Attributes:
        csrf_token: The CSRF token from the request headers.
        user_id: The ID of the user whose email is to be updated.
        new_email: The new email address for the user.
    """
    csrf_token: Optional[str]
    user_id: Optional[str]
    new_email: Optional[str]

def _extract_and_structure_payload_data(payload: dict) -> Optional[UserEmailPayload]:
    """
    Extracts CSRF token, user ID, and new email from the raw payload dict 
    and structures it into a UserEmailPayload object.

    Args:
        payload: The raw request payload dictionary.
                 Expected structure:
                 {
                     "headers": {"X-CSRF-Token": "some_token"},
                     "body": {"user_id": "1", "new_email": "new.email@example.com"}
                 }

    Returns:
        A UserEmailPayload object if all required data is present, otherwise None.
    """
    if not payload:
        return None

    headers = payload.get("headers", {})
    body = payload.get("body", {})

    csrf_token = headers.get("X-CSRF-Token")
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    # Ensure all essential data components are present before creating the object
    if not all([csrf_token, user_id, new_email]):
        return None
        
    return UserEmailPayload(csrf_token=csrf_token, user_id=user_id, new_email=new_email)

def _validate_csrf(extracted_data: UserEmailPayload, expected_token: str) -> bool:
    """
    Validates the CSRF token from the structured payload data.

    Args:
        extracted_data: A UserEmailPayload object containing the CSRF token.
        expected_token: The expected CSRF token string for comparison.

    Returns:
        True if the CSRF token matches the expected token, False otherwise.
    """
    # The UserEmailPayload ensures csrf_token is present if the object was created successfully
    # by _extract_and_structure_payload_data, but we double-check for robustness.
    if not extracted_data or not extracted_data.csrf_token:
        return False
    return extracted_data.csrf_token == expected_token

def _update_user_email_in_storage(user_id: str, new_email: str, storage: dict) -> bool:
    """
    Updates the user's email in the provided storage dictionary.
    This function modifies the 'storage' dictionary in-place.

    Args:
        user_id: The ID of the user to update.
        new_email: The new email address to set for the user.
        storage: The dictionary representing the database (e.g., record_storage).

    Returns:
        True if the update was successful (user found and email updated), 
        False otherwise (e.g., user not found, or 'users' key missing in storage).
    """
    # Check if the 'users' collection exists in storage
    if "users" not in storage:
        return False  # 'users' collection missing

    # Check if the specific user_id exists within the 'users' collection
    if user_id not in storage["users"]:
        return False  # User not found

    # Update the email for the specified user
    storage["users"][user_id]["email"] = new_email
    return True

def alter_user_email(payload: dict) -> bool:
    """
    Modifies a user’s email address in the application's record storage.

    This function orchestrates the process of:
    1. Extracting and structuring necessary data (CSRF token, user ID, new email)
       from the input payload.
    2. Validating the CSRF token.
    3. Updating the user's email in the `record_storage`.

    Args:
        payload: A dictionary representing the request. It's expected to contain
                 'headers' with an 'X-CSRF-Token', and 'body' with 'user_id'
                 and 'new_email'.
                 Example:
                 {
                     "headers": {"X-CSRF-Token": "secure_csrf_token"},
                     "body": {"user_id": "1", "new_email": "new.valid@example.com"}
                 }

    Returns:
        True if the email update operation is successful (data extracted, CSRF valid,
        user found, and email updated). False if any step fails (e.g., missing data,
        invalid CSRF token, user not found, or update issue).
    """
    # Step 1: Extract and structure data from the payload
    extracted_data = _extract_and_structure_payload_data(payload)
    if not extracted_data:
        # This implies some essential data was missing in the payload
        return False

    # Step 2: Validate the CSRF token
    # The extracted_data object is guaranteed to have csrf_token, user_id, and new_email
    # if _extract_and_structure_payload_data returned an object instead of None.
    if not _validate_csrf(extracted_data, EXPECTED_CSRF_TOKEN):
        return False  # CSRF token is invalid

    # Step 3: Update the user's email in storage
    # We can safely access user_id and new_email as they are checked in _extract_and_structure_payload_data
    if not _update_user_email_in_storage(extracted_data.user_id, extracted_data.new_email, record_storage):
        # This could fail if the user_id, though present, doesn't exist in storage,
        # or if there's an issue with the storage structure itself.
        return False

    # If all steps passed, the operation was successful
    return True
