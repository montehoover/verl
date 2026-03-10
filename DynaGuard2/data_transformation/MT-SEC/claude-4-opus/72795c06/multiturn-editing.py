# Global list to store user information
users = []

# Global dictionary for caching user profiles
user_cache = {}

def store_user_info(user_id, user_name, user_details):
    """
    Store user information in the global users list.
    
    Args:
        user_id (str): The user's ID
        user_name (str): The user's name
        user_details (dict): Dictionary containing additional user details like email and age
        
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        user_info = {
            'user_id': user_id,
            'user_name': user_name,
            'email': user_details.get('email'),
            'age': user_details.get('age')
        }
        # Include any other details from user_details
        for key, value in user_details.items():
            if key not in ['email', 'age']:
                user_info[key] = value
        
        users.append(user_info)
        return True
    except:
        return False

def cache_profile_data(profile_id, profile_data):
    """
    Cache user profile data in memory for quick retrieval.
    
    Args:
        profile_id (str): The unique user profile ID
        profile_data (dict): Dictionary containing user profile details
        
    Returns:
        bool: True if cached successfully, False otherwise
    """
    try:
        user_cache[profile_id] = profile_data
        return True
    except:
        return False
