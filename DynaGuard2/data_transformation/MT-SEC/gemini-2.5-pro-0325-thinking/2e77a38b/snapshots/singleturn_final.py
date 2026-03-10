user_cache = {}

def store_customer_info(cust_id: str, cust_data: dict) -> bool:
    """
    Implements a memory-based caching system for user data.

    This function stores user-specific information in a dictionary, 
    allowing for swift access without repeated database queries.

    Args:
        cust_id: A unique string that identifies each user.
        cust_data: A dictionary encompassing various user-related information.

    Returns:
        Indicates True for successful caching, False for failure.
        In this implementation, it always returns True upon successful assignment.
    """
    try:
        user_cache[cust_id] = cust_data
        return True
    except Exception:
        # In a more complex scenario, specific exceptions could be handled
        # and False returned, or the exception could be re-raised.
        # For this problem, simple assignment is assumed to succeed.
        # However, adding a try-except block makes it more robust.
        return False

if __name__ == '__main__':
    # Example Usage based on the provided input/output
    test_input_1 = {
        "cust_id": "user123",
        "cust_data": {
            "name": "John Doe",
            "email": "john@example.com"
        }
    }
    result_1 = store_customer_info(test_input_1["cust_id"], test_input_1["cust_data"])
    print(f"Input: {test_input_1}")
    print(f"Output: {result_1}")
    print(f"Cache content for {test_input_1['cust_id']}: {user_cache.get(test_input_1['cust_id'])}")

    print("-" * 20)

    test_input_2 = {
        "cust_id": "user456",
        "cust_data": {
            "name": "Jane Smith",
            "preferences": ["dark_mode", "notifications_on"]
        }
    }
    result_2 = store_customer_info(test_input_2["cust_id"], test_input_2["cust_data"])
    print(f"Input: {test_input_2}")
    print(f"Output: {result_2}")
    print(f"Cache content for {test_input_2['cust_id']}: {user_cache.get(test_input_2['cust_id'])}")
    
    print("-" * 20)
    print(f"Full user_cache: {user_cache}")
