USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

# Global data structure to store project memberships
# Example: {project_id_1: {user_id_1, user_id_2}, project_id_2: {user_id_2, user_id_3}}
project_memberships = {}

def is_member_of_project(user_id: int, project_id: int) -> bool:
    """
    Checks if a specific user is a member of a given project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.

    Returns:
        True if the user is part of the project, otherwise False.
    """
    if project_id in project_memberships:
        return user_id in project_memberships[project_id]
    return False

def add_user_to_project(user_id: int, project_id: int) -> None:
    """
    Adds a user to a specific project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.
    """
    if project_id not in project_memberships:
        project_memberships[project_id] = set()
    project_memberships[project_id].add(user_id)

def remove_user_from_project(user_id: int, project_id: int) -> None:
    """
    Removes a user from a specific project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.
    """
    if project_id in project_memberships:
        project_memberships[project_id].discard(user_id)

def count_active_projects_for_user(user_id: int) -> int:
    """
    Counts how many active projects a user is a member of.

    Args:
        user_id: The ID of the user.

    Returns:
        The number of projects the user is a member of.
    """
    count = 0
    for project_id in project_memberships:
        if user_id in project_memberships[project_id]:
            count += 1
    return count

def update_project_members(acting_user_id: int, project_identifier: int, modification: str, user_to_update_id: int) -> bool:
    """
    Allows project owners or admins to add or remove team members from projects.

    Args:
        acting_user_id: The ID of the user performing the action.
        project_identifier: The ID of the project to modify.
        modification: The action to perform ('add' or 'remove').
        user_to_update_id: The ID of the user to add or remove.

    Returns:
        True if the modification was successful, False otherwise.
    """
    if project_identifier not in PROJECTS:
        return False  # Project does not exist

    if acting_user_id not in USERS:
        return False  # Acting user does not exist

    project = PROJECTS[project_identifier]
    acting_user = USERS[acting_user_id]

    is_admin = acting_user['role'] == 'admin'
    is_owner = project['owner_id'] == acting_user_id

    if not (is_admin or is_owner):
        return False  # Not authorized to modify this project

    if modification == 'add':
        if user_to_update_id not in project['members']:
            # Optional: Check if user_to_update_id exists in USERS before adding.
            # if user_to_update_id not in USERS:
            #     return False # User to add does not exist in the system
            project['members'].append(user_to_update_id)
        # Modification is considered successful if 'add' is requested,
        # regardless of whether the user was already a member.
        return True
    elif modification == 'remove':
        if user_to_update_id in project['members']:
            # Optional: Add checks here if an owner cannot remove themselves,
            # or if the last member cannot be removed, etc.
            # Example: if project['owner_id'] == user_to_update_id and user_to_update_id == acting_user_id:
            # return False # Owner cannot remove themselves
            project['members'].remove(user_to_update_id)
            return True  # Successfully removed
        else:
            # User to remove was not a member, so modification is not successful in terms of state change.
            return False
    else:
        return False  # Invalid modification type
