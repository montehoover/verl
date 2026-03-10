# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def get_user_details(user_request):
    user_id = user_request.get("body", {}).get("user_id")
    if user_id in database["users"]:
        return database["users"][user_id]["email"]
    return None
