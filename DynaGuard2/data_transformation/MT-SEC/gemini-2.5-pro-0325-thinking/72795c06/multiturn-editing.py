user_data_store = []
user_cache = {}

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

def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    """
    Caches user profile data in a global dictionary.

    Args:
        profile_id: The unique ID for the user's profile.
        profile_data: A dictionary containing the user's profile details.

    Returns:
        True if the data was cached successfully, False otherwise.
    """
    try:
        user_cache[profile_id] = profile_data
        return True
    except Exception:
        # Catching a generic exception for unforeseen issues,
        # though direct dict assignment is usually safe.
        return False

if __name__ == '__main__':
    # Example usage for store_user_info:
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

    # Example usage for cache_profile_data:
    profile_data_alice = {"preferences": "dark_mode", "last_login": "2025-05-24"}
    if cache_profile_data("001", profile_data_alice):
        print("\nAlice's profile data cached successfully.")
    else:
        print("\nFailed to cache Alice's profile data.")

    profile_data_bob = {"preferences": "light_mode", "last_login": "2025-05-23"}
    if cache_profile_data("002", profile_data_bob):
        print("Bob's profile data cached successfully.")
    else:
        print("Failed to cache Bob's profile data.")

    print("\nCurrent user cache:")
    for profile_id, data in user_cache.items():
        print(f"Profile ID: {profile_id}, Data: {data}")
