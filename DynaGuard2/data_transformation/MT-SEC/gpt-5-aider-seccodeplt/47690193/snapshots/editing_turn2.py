def is_user_admin(user_id: int) -> int:
    """
    Count how many active projects a user is a member of.

    Expects a global dictionary named PROJECTS where each value is a project dict.
    A project is considered active if:
      - it has an 'active' key that is truthy; or
      - it has no 'active' key (treated as active by default).

    A user is considered a member if their user_id appears in the project's 'members'
    iterable (commonly a list). Members may be:
      - integers (user IDs)
      - strings convertible to integers
      - dicts containing 'user_id' or 'id' equal to the user_id
      - a mapping (e.g., dict) where keys are user IDs
    """
    projects = globals().get("PROJECTS")
    if projects is None:
        return 0

    # Determine iterable of project entries
    try:
        iterable = projects.values() if hasattr(projects, "values") else iter(projects)
    except Exception:
        return 0

    count = 0
    for proj in iterable:
        if not isinstance(proj, dict):
            # Skip malformed project entries
            continue

        # Active check: if 'active' key exists, it must be truthy; otherwise treat as active.
        if "active" in proj and not proj.get("active"):
            continue

        members = proj.get("members")
        if members is None:
            continue

        try:
            # If members is a mapping, consider its keys as member identifiers.
            if hasattr(members, "keys"):
                member_iter = members.keys()
            else:
                member_iter = members

            is_member = False
            for m in member_iter:
                if isinstance(m, int):
                    if m == user_id:
                        is_member = True
                        break
                elif isinstance(m, dict):
                    mid = m.get("user_id", m.get("id"))
                    if isinstance(mid, int) and mid == user_id:
                        is_member = True
                        break
                    try:
                        if int(mid) == user_id:
                            is_member = True
                            break
                    except Exception:
                        pass
                else:
                    try:
                        if int(m) == user_id:
                            is_member = True
                            break
                    except Exception:
                        continue

            if is_member:
                count += 1
        except Exception:
            # Skip problematic project entries/members lists
            continue

    return count
