import re

# Mock database of users and their phone numbers
MOCK_USER_DATABASE = {
    "existing_user_valid_phone": {"name": "Alice", "phone": "123-456-7890"},
    "existing_user_invalid_phone": {"name": "Bob", "phone": "1234567890"},
    "existing_user_no_phone": {"name": "Charlie"},
}

def _is_valid_phone_format(phone_number: str) -> bool:
    """Checks if the phone number is in XXX-XXX-XXXX format."""
    if phone_number is None:
        return False # Or True, depending on whether a missing phone is considered valid by default
    pattern = r"^\d{3}-\d{3}-\d{4}$"
    return bool(re.match(pattern, phone_number))

def check_user_exists_and_phone_valid(user_id: str) -> bool:
    """
    Verifies if a user exists in a database and their phone number is valid.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists and phone number is valid, False otherwise.
        Prints a message if the phone number format is invalid.
    """
    print(f"Checking user '{user_id}'...")

    user_data = MOCK_USER_DATABASE.get(user_id)

    if not user_data:
        print(f"User '{user_id}' does not exist.")
        return False

    print(f"User '{user_id}' exists.")
    phone_number = user_data.get("phone")

    if phone_number is None:
        print(f"User '{user_id}' does not have a phone number.")
        # Decide if this case means the check fails or passes.
        # For this example, let's say it fails if a phone number is expected.
        return False

    if not _is_valid_phone_format(phone_number):
        print(f"Invalid phone number format for user '{user_id}': {phone_number}. Expected XXX-XXX-XXXX.")
        return False

    print(f"User '{user_id}' has a valid phone number: {phone_number}.")
    return True

if __name__ == '__main__':
    # Example usage:
    print("\n--- Test Case 1: User exists, valid phone ---")
    user1_valid = check_user_exists_and_phone_valid("existing_user_valid_phone")
    print(f"Check result for 'existing_user_valid_phone': {user1_valid}\n")

    print("--- Test Case 2: User exists, invalid phone ---")
    user2_valid = check_user_exists_and_phone_valid("existing_user_invalid_phone")
    print(f"Check result for 'existing_user_invalid_phone': {user2_valid}\n")

    print("--- Test Case 3: User exists, no phone ---")
    user3_valid = check_user_exists_and_phone_valid("existing_user_no_phone")
    print(f"Check result for 'existing_user_no_phone': {user3_valid}\n")

    print("--- Test Case 4: User does not exist ---")
    user4_valid = check_user_exists_and_phone_valid("non_existing_user")
    print(f"Check result for 'non_existing_user': {user4_valid}")
