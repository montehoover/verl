# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def get_user_phone(user_id: str) -> str | None:
    """
    Retrieves the phone number for a given user_id from the simulated database.

    Args:
        user_id: The ID of the user.

    Returns:
        The user's phone number if the user exists, otherwise None.
    """
    if user_id in database["users"]:
        return database["users"][user_id]["phone"]
    return None

if __name__ == '__main__':
    # Example usage:
    phone1 = get_user_phone('1')
    if phone1:
        print(f"User '1' phone: {phone1}")
    else:
        print(f"User '1' not found.")

    phone2 = get_user_phone('2')
    if phone2:
        print(f"User '2' phone: {phone2}")
    else:
        print(f"User '2' not found.")
