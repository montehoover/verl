# Global data structure to track user roles
user_roles = {}

def check_user_role(user_id):
    """Determines if a user is an admin or a regular user.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        str: 'admin' if the user is an admin, 'user' if they are a regular user
    """
    return user_roles.get(user_id, 'user')

def format_project_title(title):
    """Formats a project title in title case with common stop words lowercase.
    
    Args:
        title (str): The project title to format
        
    Returns:
        str: The formatted title with proper capitalization
    """
    stop_words = {'and', 'of', 'in', 'the', 'a', 'an', 'to', 'for', 'with', 'on', 'at', 'by', 'or'}
    words = title.split()
    
    formatted_words = []
    for i, word in enumerate(words):
        # Always capitalize the first word, otherwise check if it's a stop word
        if i == 0 or word.lower() not in stop_words:
            formatted_words.append(word.capitalize())
        else:
            formatted_words.append(word.lower())
    
    return ' '.join(formatted_words)
