activity_log = []

def count_user_activities(user_alias: str) -> int:
    """
    Counts the number of recorded actions for a given user from the global activity log.

    Args:
        user_alias: The identifier string for the user.

    Returns:
        An integer representing the number of actions logged by the user.
    """
    activity_count = 0
    for entry in activity_log:
        if entry.get('user_alias') == user_alias:
            activity_count += 1
    return activity_count
