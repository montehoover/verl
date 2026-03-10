def is_user_admin(user_id: int) -> int:
    """
    Return the number of projects in the global PROJECTS dictionary where the given
    user_id appears in the project's 'members' list.

    Expects a global data structure named PROJECTS that maps project IDs to dicts,
    each possibly containing a 'members' iterable (e.g., list, set, tuple) of user IDs.
    """
    projects = globals().get('PROJECTS', {})
    if not isinstance(projects, dict):
        return 0

    count = 0
    for project in projects.values():
        if not isinstance(project, dict):
            continue
        members = project.get('members', [])
        try:
            if any(member == user_id for member in members):
                count += 1
        except TypeError:
            # members is not iterable
            continue

    return count


def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Allows project owners to add/remove team members from their own projects,
    while admins can manage any project.

    Arguments:
        executor_user_id: ID of the user performing the operation.
        proj_id: Project ID to modify.
        operation: 'add' or 'remove'.
        target_member_id: ID of the user to add/remove.

    Returns:
        True if the modification is successful, otherwise False.

    Expects global dicts:
        USERS: { user_id: { "user_id": int, "role": "admin"|"user", ... }, ... }
        PROJECTS: {
            project_id: { "owner_id": int, "members": [int, ...], ... },
            ...
        }
    """
    users = globals().get('USERS', {})
    projects = globals().get('PROJECTS', {})
    if not isinstance(users, dict) or not isinstance(projects, dict):
        return False

    # Validate executor and project
    executor = users.get(executor_user_id)
    project = projects.get(proj_id)
    if not isinstance(executor, dict) or not isinstance(project, dict):
        return False

    # Authorization: admin can manage any project; owner can manage their own project
    role = str(executor.get('role', '')).lower()
    is_admin = role == 'admin'
    is_owner = project.get('owner_id') == executor_user_id
    if not (is_admin or is_owner):
        return False

    # Validate target user exists
    if target_member_id not in users:
        return False

    op = str(operation).lower()
    if op not in ('add', 'remove'):
        return False

    # Normalize members to a list in the project
    members = project.get('members')
    if not isinstance(members, list):
        try:
            members = list(members) if members is not None else []
        except TypeError:
            members = []
        project['members'] = members

    if op == 'add':
        if target_member_id in members:
            return False
        members.append(target_member_id)
        return True

    # op == 'remove'
    try:
        members.remove(target_member_id)
        return True
    except ValueError:
        return False
