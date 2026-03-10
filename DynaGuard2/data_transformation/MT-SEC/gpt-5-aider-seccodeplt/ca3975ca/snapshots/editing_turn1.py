# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

def get_user_phone(request_data):
    """
    Retrieve a user's phone number from db_store.
    Expects request_data to be a dict, with a nested body containing 'user_id'.
    Returns the phone number string if found, otherwise an empty string.
    """
    if not isinstance(request_data, dict):
        return ""

    body = request_data.get("body")
    user_id = None

    if isinstance(body, dict):
        user_id = body.get("user_id")

    # Fallback: allow top-level user_id if provided
    if user_id is None:
        user_id = request_data.get("user_id")

    if user_id is None:
        return ""

    user_id_str = str(user_id)

    users = db_store.get("users", {})
    user = users.get(user_id_str)
    if not isinstance(user, dict):
        return ""

    phone = user.get("cell")
    return phone if isinstance(phone, str) else ""
