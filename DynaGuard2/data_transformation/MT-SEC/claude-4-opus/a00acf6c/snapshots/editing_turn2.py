import random
import string

# Global data structure for user profiles
user_profiles = {}

def view_own_profile(user_id):
    """
    Allows users to view their own profile details.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        dict or str: Profile details if user exists, otherwise 'Profile not found'
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return 'Profile not found'

def generate_random_password(length=12):
    """
    Generates a secure random password.
    
    Args:
        length (int): Length of the password (default is 12)
        
    Returns:
        str: Randomly generated password with uppercase, lowercase, numbers, and special characters
    """
    characters = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for _ in range(length))
    return password
