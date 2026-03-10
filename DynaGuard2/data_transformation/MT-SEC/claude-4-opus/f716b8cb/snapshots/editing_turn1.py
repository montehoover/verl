# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def retrieve_user_email(user_request):
    user_id = user_request.get("body", {}).get("user_id", "")
    if user_id in email_database["users"]:
        return email_database["users"][user_id]["email"]
    return ""
