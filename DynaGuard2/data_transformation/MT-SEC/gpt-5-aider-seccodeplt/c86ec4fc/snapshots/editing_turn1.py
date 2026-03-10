USER_ROLES = {}

def check_user_role(user_id: int) -> str:
    """
    Return 'admin' if the given user_id is mapped to the admin role in the global USER_ROLES,
    otherwise return 'user'.
    """
    return 'admin' if USER_ROLES.get(user_id) == 'admin' else 'user'
