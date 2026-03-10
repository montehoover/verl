activity_log = []

def get_user_activities(user_alias: str) -> list:
    """
    Retrieves a list of all recorded actions for a given user from the global activity log.

    Args:
        user_alias: The identifier string for the user.

    Returns:
        A list of interaction descriptions related to the specified user.
    """
    user_interactions = []
    for entry in activity_log:
        if entry.get('user_alias') == user_alias:
            user_interactions.append(entry.get('interaction_desc'))
    return user_interactions
