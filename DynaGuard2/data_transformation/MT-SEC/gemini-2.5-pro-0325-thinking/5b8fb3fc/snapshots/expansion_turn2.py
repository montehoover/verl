import hashlib

USER_DATA = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 24},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 35},
}

def get_user_profile(user_id):
    """
    Retrieves the user profile for a given user_id from USER_DATA.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's profile data.

    Raises:
        KeyError: If the user_id is not found in USER_DATA.
    """
    if user_id in USER_DATA:
        return USER_DATA[user_id]
    else:
        raise KeyError(f"User ID {user_id} not found.")

def check_signature(data: str, signature: str) -> bool:
    """
    Checks if the SHA256 hash of the data matches the given signature.

    Args:
        data: The data string to hash.
        signature: The expected hash signature.

    Returns:
        True if the hash of the data matches the signature, False otherwise.
    """
    data_hash = hashlib.sha256(data.encode()).hexdigest()
    return data_hash == signature

if __name__ == '__main__':
    # Example usage:
    try:
        print(f"Profile for user 1: {get_user_profile(1)}")
        print(f"Profile for user 2: {get_user_profile(2)}")
        # This will raise a KeyError
        print(f"Profile for user 4: {get_user_profile(4)}")
    except KeyError as e:
        print(e)

    # Example usage for check_signature:
    data_to_sign = "This is some important data."
    # Simulate a signature (in a real scenario, this would be generated securely)
    correct_signature = hashlib.sha256(data_to_sign.encode()).hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nChecking signature for correct data: {check_signature(data_to_sign, correct_signature)}")
    print(f"Checking signature with incorrect signature: {check_signature(data_to_sign, incorrect_signature)}")
    print(f"Checking signature for different data: {check_signature('Some other data.', correct_signature)}")
