import hashlib

# Example local user database
LOCAL_USER_DATABASE = {
    "user123": {
        "name": "Alice Wonderland",
        "email": "alice@example.com",
        "preferences": {
            "theme": "dark",
            "notifications": "email_only"
        }
    },
    "user456": {
        "name": "Bob The Builder",
        "email": "bob@example.com",
        "preferences": {
            "theme": "light",
            "notifications": "all"
        }
    },
    "user789": {
        "name": "Charlie Brown",
        "email": "charlie@example.com",
        "preferences": {
            "theme": "system",
            "notifications": "none"
        }
    }
}

def get_local_user_profile(user_id: str) -> dict:
    """
    Retrieves a user's profile from the local database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's profile data.

    Raises:
        KeyError: If the user_id does not exist in the database.
    """
    if user_id in LOCAL_USER_DATABASE:
        return LOCAL_USER_DATABASE[user_id]
    else:
        raise KeyError(f"User ID '{user_id}' not found in local database.")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates a signature against the given data using SHA256.

    Args:
        data: The data string that was signed.
        signature: The expected signature (hex digest).

    Returns:
        True if the signature matches the data's hash, False otherwise.
    """
    hasher = hashlib.sha256()
    hasher.update(data.encode('utf-8'))
    computed_signature = hasher.hexdigest()
    return computed_signature == signature

if __name__ == '__main__':
    # Example usage:
    try:
        profile = get_local_user_profile("user123")
        print(f"Profile for user123: {profile}")
    except KeyError as e:
        print(e)

    try:
        profile = get_local_user_profile("user999") # This user does not exist
        print(f"Profile for user999: {profile}")
    except KeyError as e:
        print(e)

    # Example usage for validate_signature:
    data_to_sign = "This is some important data."
    # Simulate a signature (in a real scenario, this would be generated securely)
    hasher = hashlib.sha256()
    hasher.update(data_to_sign.encode('utf-8'))
    correct_signature = hasher.hexdigest()
    incorrect_signature = "thisisnotthesignature"

    print(f"Validating correct signature: {validate_signature(data_to_sign, correct_signature)}")
    print(f"Validating incorrect signature: {validate_signature(data_to_sign, incorrect_signature)}")
