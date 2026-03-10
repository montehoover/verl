def get_user_info(user_id):
    # Mock database - in a real application, this would query an actual database
    database = {
        "user123": "john.doe@example.com",
        "user456": "jane.smith@example.com",
        "user789": "bob.johnson@example.com"
    }
    
    return database.get(user_id)
