user_data_store = []

def store_user_info(user_id: str, user_name: str, user_details: dict) -> bool:
    """
    Stores user information, including additional details, into a global list.

    Args:
        user_id: The ID of the user.
        user_name: The name of the user.
        user_details: A dictionary containing additional user details like email and age.

    Returns:
        True if the information was stored successfully, False otherwise.
    """
    try:
        user_data_store.append({
            "user_id": user_id,
            "user_name": user_name,
            "details": user_details
        })
        return True
    except Exception:
        # In a simple list append, an exception is highly unlikely
        # unless there are severe memory issues.
        return False

if __name__ == '__main__':
    # Example usage:
    alice_details = {"email": "alice@example.com", "age": 30}
    if store_user_info("001", "Alice", alice_details):
        print("User Alice stored successfully.")
    else:
        print("Failed to store Alice.")

    bob_details = {"email": "bob@example.com", "age": 25}
    if store_user_info("002", "Bob", bob_details):
        print("User Bob stored successfully.")
    else:
        print("Failed to store Bob.")

    print("\nCurrent user data:")
    for user in user_data_store:
        print(user)
