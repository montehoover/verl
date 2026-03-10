USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def get_user_info(user_id: int) -> dict | None:
    """
    Retrieves user details given a user ID.

    Args:
        user_id: The ID of the user.

    Returns:
        A dictionary with the user's role and privilege level,
        or None if the user is not found.
    """
    user_data = USERS.get(user_id)
    if user_data:
        return {
            "role": user_data["role"],
            "privilege": user_data["privilege"]
        }
    return None

def can_change_privilege(actor_id: int) -> bool:
    """
    Checks if a user has the authority to modify others' privilege levels.

    Args:
        actor_id: The ID of the user attempting the action.

    Returns:
        True if the user can change privileges, False otherwise.
    """
    actor_info = get_user_info(actor_id)
    if actor_info and actor_info["role"] == "admin":
        return True
    return False

def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Updates a user's privilege level if the actor has permission.

    Args:
        actor_id: The ID of the user attempting the action.
        recipient_id: The ID of the user whose privilege is to be updated.
        updated_privilege: The new privilege level ('normal', 'moderator', 'admin').

    Returns:
        True if the privilege was updated successfully, False otherwise.
    """
    if not can_change_privilege(actor_id):
        print(f"User ID {actor_id} does not have permission to change privileges.")
        return False

    if recipient_id not in USERS:
        print(f"Recipient User ID {recipient_id} not found.")
        return False

    valid_privileges = ["normal", "moderator", "admin"]
    if updated_privilege not in valid_privileges:
        print(f"Invalid privilege level: {updated_privilege}. Must be one of {valid_privileges}.")
        return False

    USERS[recipient_id]["privilege"] = updated_privilege
    print(f"User ID {recipient_id}'s privilege updated to {updated_privilege} by User ID {actor_id}.")
    return True

if __name__ == '__main__':
    # Example usage:
    user_id_to_find = 2
    user_info = get_user_info(user_id_to_find)
    if user_info:
        print(f"User ID {user_id_to_find}: Role - {user_info['role']}, Privilege - {user_info['privilege']}")
    else:
        print(f"User ID {user_id_to_find} not found.")

    # Example usage for update_user_privilege:
    admin_user = 1
    target_user = 2
    non_admin_user = 3
    non_existent_target_user = 99

    print("\n--- Update Privilege Scenarios ---")
    # Admin attempts to update user's privilege to moderator
    print(f"\nAttempting to update User ID {target_user} privilege by Admin ID {admin_user}:")
    success = update_user_privilege(admin_user, target_user, "moderator")
    print(f"Update successful: {success}")
    if success:
        print(f"User ID {target_user} new info: {get_user_info(target_user)}")

    # Non-admin attempts to update user's privilege
    print(f"\nAttempting to update User ID {target_user} privilege by User ID {non_admin_user}:")
    success = update_user_privilege(non_admin_user, target_user, "admin")
    print(f"Update successful: {success}")
    if success: # Should not happen
        print(f"User ID {target_user} new info: {get_user_info(target_user)}")
    else:
        print(f"User ID {target_user} info remains: {get_user_info(target_user)}")


    # Admin attempts to update non-existent user's privilege
    print(f"\nAttempting to update non-existent User ID {non_existent_target_user} privilege by Admin ID {admin_user}:")
    success = update_user_privilege(admin_user, non_existent_target_user, "admin")
    print(f"Update successful: {success}")

    # Admin attempts to update user's privilege to an invalid level
    print(f"\nAttempting to update User ID {target_user} privilege to 'superadmin' by Admin ID {admin_user}:")
    success = update_user_privilege(admin_user, target_user, "superadmin")
    print(f"Update successful: {success}")
    if not success:
        print(f"User ID {target_user} info remains: {get_user_info(target_user)}")

    # Reset user 2 privilege for consistency if other tests are run
    update_user_privilege(admin_user, target_user, "normal")

    # Example usage for can_change_privilege:
    actor_id_admin = 1
    actor_id_user = 2
    actor_id_non_existent = 99

    print(f"User ID {actor_id_admin} can change privilege: {can_change_privilege(actor_id_admin)}")
    print(f"User ID {actor_id_user} can change privilege: {can_change_privilege(actor_id_user)}")
    print(f"User ID {actor_id_non_existent} can change privilege: {can_change_privilege(actor_id_non_existent)}")

    user_id_to_find = 5
    user_info = get_user_info(user_id_to_find)
    if user_info:
        print(f"User ID {user_id_to_find}: Role - {user_info['role']}, Privilege - {user_info['privilege']}")
    else:
        print(f"User ID {user_id_to_find} not found.")
    
    user_id_to_find = 1
    user_info = get_user_info(user_id_to_find)
    if user_info:
        print(f"User ID {user_id_to_find}: Role - {user_info['role']}, Privilege - {user_info['privilege']}")
    else:
        print(f"User ID {user_id_to_find} not found.")
