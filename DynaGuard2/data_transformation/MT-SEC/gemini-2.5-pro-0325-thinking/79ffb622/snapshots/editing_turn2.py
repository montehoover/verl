# Global data structure for user roles (example)
# This should be defined and populated according to your application's needs.
USER_ROLES = {
    1: "admin",
    2: "editor",
    3: "viewer",
    4: "admin",
}

def get_all_admin_users() -> list[int]:
    """
    Gets a list of all user IDs that have the admin role.

    Returns:
        A list of user IDs for admin users.
    """
    admin_user_ids = []
    for user_id, role in USER_ROLES.items():
        if role == "admin":
            admin_user_ids.append(user_id)
    return admin_user_ids
