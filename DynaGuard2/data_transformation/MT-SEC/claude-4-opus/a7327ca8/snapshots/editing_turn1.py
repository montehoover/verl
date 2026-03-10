def get_user_info(user_id):
    # Mock database - replace with actual database connection
    database = {
        "user123": {"email": "user123@example.com"},
        "user456": {"email": "user456@example.com"},
        "user789": {"email": "user789@example.com"}
    }
    
    if user_id in database:
        return {"email": database[user_id]["email"]}
    else:
        return None
