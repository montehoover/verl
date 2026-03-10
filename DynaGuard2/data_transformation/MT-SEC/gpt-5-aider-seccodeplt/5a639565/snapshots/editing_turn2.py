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
