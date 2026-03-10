# Global dictionary to store user information
users_db = {}

# Global dictionary acting as the repository for cached user data
user_cache = {}

def store_customer_info(cust_id: str, cust_data: dict) -> bool:
    """
    Stores customer information in the global user cache.

    Args:
        cust_id: The unique ID of the customer (string).
        cust_data: A dictionary containing customer-related information.

    Returns:
        True if the customer information was stored successfully, False otherwise.
    """
    try:
        user_cache[cust_id] = cust_data
        return True
    except Exception:
        # In a more complex scenario, you might have specific error handling
        return False

def add_user(user_id: str, user_info: dict) -> bool:
    """
    Adds a user to the global user dictionary.

    Args:
        user_id: The ID of the user (string).
        user_info: A dictionary containing user details (e.g., 'name', 'email').

    Returns:
        True if the user was added successfully, False otherwise.
    """
    # Check if user_id already exists to prevent overwriting,
    # or decide if overwriting is the desired behavior.
    # For now, let's allow overwriting.
    try:
        users_db[user_id] = user_info
        return True
    except Exception:
        # In a more complex scenario, you might have specific error handling
        return False

if __name__ == '__main__':
    # Example usage:
    user1_info = {"name": "Alice Wonderland", "email": "alice@example.com"}
    added1 = add_user("user123", user1_info)
    print(f"User 'user123' added: {added1}")

    user2_info = {"name": "Bob The Builder", "email": "bob@example.com"}
    added2 = add_user("user456", user2_info)
    print(f"User 'user456' added: {added2}")

    print("\nCurrent users in DB:")
    for user_id, info in users_db.items():
        print(f"ID: {user_id}, Info: {info}")

    # Example usage for store_customer_info:
    cust1_data = {"name": "Charlie Brown", "loyalty_points": 150}
    stored1 = store_customer_info("cust001", cust1_data)
    print(f"\nCustomer 'cust001' data stored in cache: {stored1}")

    cust2_data = {"name": "Lucy Van Pelt", "preferences": ["psychiatry", "complaining"]}
    stored2 = store_customer_info("cust002", cust2_data)
    print(f"Customer 'cust002' data stored in cache: {stored2}")

    print("\nCurrent customer info in cache:")
    for cust_id, data in user_cache.items():
        print(f"Customer ID: {cust_id}, Data: {data}")
