user_data_store = []

def store_user_info(user_id: str, user_name: str) -> bool:
    """
    Stores user information into a global list.

    Args:
        user_id: The ID of the user.
        user_name: The name of the user.

    Returns:
        True if the information was stored successfully, False otherwise.
    """
    try:
        user_data_store.append({"user_id": user_id, "user_name": user_name})
        return True
    except Exception:
        # In a simple list append, an exception is highly unlikely
        # unless there are severe memory issues.
        return False

if __name__ == '__main__':
    # Example usage:
    if store_user_info("001", "Alice"):
        print("User Alice stored successfully.")
    else:
        print("Failed to store Alice.")

    if store_user_info("002", "Bob"):
        print("User Bob stored successfully.")
    else:
        print("Failed to store Bob.")

    print("\nCurrent user data:")
    for user in user_data_store:
        print(user)
