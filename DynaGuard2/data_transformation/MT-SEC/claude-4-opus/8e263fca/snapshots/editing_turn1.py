def get_local_profile(user_ident):
    # Local dictionary of user profiles
    user_profiles = {
        "user123": {"name": "John Doe", "email": "john.doe@example.com"},
        "user456": {"name": "Jane Smith", "email": "jane.smith@example.com"},
        "user789": {"name": "Bob Johnson", "email": "bob.johnson@example.com"}
    }
    
    # Return the user profile if found, otherwise return None
    return user_profiles.get(user_ident)
