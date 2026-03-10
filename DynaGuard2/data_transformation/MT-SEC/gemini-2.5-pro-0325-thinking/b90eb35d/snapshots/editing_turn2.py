# Global data structure for user roles and privileges
# This is a placeholder and should be replaced with your actual data structure
USER_PRIVILEGES = {
    1: "admin",
    2: "editor",
    3: "viewer",
    # Add more users and their privileges here
}

def get_user_privilege(user_id: int) -> str:
    """
    Checks the current privilege level of a user.

    Args:
        user_id: The ID of the user.

    Returns:
        The privilege level of the user as a string, 
        or "unknown" if the user is not found.
    """
    return USER_PRIVILEGES.get(user_id, "unknown")

def generate_user_report() -> list:
    """
    Generates a list of all user IDs and their privilege levels.

    Returns:
        A list of dictionaries, where each dictionary contains
        'user_id' and 'privilege'.
    """
    report = []
    for user_id, privilege in USER_PRIVILEGES.items():
        report.append({'user_id': user_id, 'privilege': privilege})
    return report
