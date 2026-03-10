activity_log = []

# Placeholder for potential future configuration
MAX_LOG_ENTRIES = None  # Or some integer value like 1000


def _create_log_entry(user_alias: str, interaction_desc: str) -> dict:
    """Create a structured log entry.

    This helper function encapsulates the creation of a log entry dictionary,
    allowing for easy modification of the log entry structure in one place.
    Future enhancements could include adding timestamps or more detailed metadata.

    Args:
        user_alias: The unique identifier for the user.
        interaction_desc: A textual description of the user's interaction.

    Returns:
        A dictionary representing the log entry, containing the user alias
        and interaction description.
        For example:
        {
            "user_alias": "user123",
            "interaction_desc": "Clicked the submit button"
        }
    """
    # Example of a more detailed log entry with a timestamp:
    # from datetime import datetime
    # return {
    #     "timestamp": datetime.utcnow().isoformat(),
    #     "user_alias": user_alias,
    #     "interaction_desc": interaction_desc
    # }
    return {"user_alias": user_alias, "interaction_desc": interaction_desc}


def _can_add_entry(log: list, max_entries: int = None) -> bool:
    """Check if a new entry can be added to the log based on defined limits.

    This helper function determines if the activity log has space for new entries
    according to the `max_entries` limit. If `max_entries` is None,
    it implies no limit is enforced.

    Args:
        log: The current activity log (list of entries).
        max_entries: The maximum number of entries allowed in the log.
                     If None, no limit check is performed.

    Returns:
        True if a new entry can be added (either no limit or limit not reached),
        False if the log is full.
    """
    if max_entries is not None and len(log) >= max_entries:
        return False  # Log limit exceeded
    return True


def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """Record a user interaction in the activity log.

    This function serves as the primary interface for logging user actions.
    It orchestrates the creation of a log entry and its addition to the
    global `activity_log`, provided that logging conditions (e.g., log size limits)
    are met.

    The function is designed for monitoring user behavior, generating audit trails,
    or feeding into analytics reports.

    Args:
        user_alias: A unique identifier for the user who performed the action.
                    (e.g., "john.doe", "user_007").
        interaction_desc: A textual description of the action taken by the user.
                          (e.g., "Logged in", "Viewed page /products/123").

    Returns:
        True if the log entry was successfully created and added to the activity_log.
        False if the log entry was not added, for instance, due to the
        activity_log exceeding a predefined `MAX_LOG_ENTRIES` limit.
    """
    if not _can_add_entry(activity_log, MAX_LOG_ENTRIES):
        return False  # Log limit exceeded or other pre-condition failed

    log_entry = _create_log_entry(user_alias, interaction_desc)
    activity_log.append(log_entry)
    return True
