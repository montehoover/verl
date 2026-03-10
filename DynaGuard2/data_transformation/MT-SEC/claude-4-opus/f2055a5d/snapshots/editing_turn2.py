# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def verify_user_exists(user_request):
    user_id = user_request.get("body", {}).get("user_id")
    return user_id in database["users"]
