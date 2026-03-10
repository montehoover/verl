def is_user_admin(user_id: int) -> int:
    """
    Count how many projects the user with the given user_id is a member of.

    Expects a global data structure PROJECTS (a mapping/dict) where each value is a
    project record. Each project record should contain:
      - 'members': an iterable of member identifiers (typically user_id integers).
        Optionally, members can be dicts containing 'user_id' or 'id'.
      - optionally 'active': if present and falsy, the project is considered inactive
        and is not counted.

    Args:
        user_id: The user's integer identifier.

    Returns:
        The number of active projects in PROJECTS where the user appears in the 'members' list.
        Returns 0 if PROJECTS is not defined, not a mapping, or the user is not in any project.
    """
    if not isinstance(user_id, int):
        return 0

    # Access the global projects structure if available.
    try:
        projects = PROJECTS  # type: ignore[name-defined]
    except NameError:
        return 0

    # Ensure we can iterate over project records.
    try:
        project_values = projects.values()  # type: ignore[attr-defined]
    except AttributeError:
        return 0

    count = 0
    for proj in project_values:
        # Skip non-dict project entries
        if not isinstance(proj, dict):
            continue

        # If an 'active' flag exists and is falsy, skip this project.
        if "active" in proj and not proj.get("active"):
            continue

        members = proj.get("members")
        if members is None:
            continue

        # Ensure members is iterable
        try:
            iterator = iter(members)
        except TypeError:
            continue

        found = False
        for m in iterator:
            if m == user_id:
                found = True
                break
            if isinstance(m, dict):
                if m.get("user_id") == user_id or m.get("id") == user_id:
                    found = True
                    break

        if found:
            count += 1

    return count
