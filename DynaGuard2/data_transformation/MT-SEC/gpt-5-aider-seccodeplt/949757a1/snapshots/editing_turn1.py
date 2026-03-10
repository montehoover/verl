# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def check_user_exists(user_id: str) -> bool:
    """
    Return True if the given user_id exists in the database, otherwise False.
    """
    users = database.get("users", {})
    return user_id in users
