import datetime

activity_log = []

def log_user_event(user_key: str, action_details: str) -> bool:
    """
    Logs a user action to the activity_log list.

    Args:
        user_key: The unique identifier for the user.
        action_details: A description of the action performed by the user.

    Returns:
        True if the event was logged successfully.
    """
    timestamp = datetime.datetime.now()
    event_record = {
        "timestamp": timestamp,
        "user_key": user_key,
        "action_details": action_details
    }
    activity_log.append(event_record)
    return True

if __name__ == '__main__':
    # Example usage:
    if log_user_event("user123", "Logged in"):
        print("User event 'Logged in' for user123 logged.")
    else:
        print("Failed to log user event.")

    if log_user_event("user456", "Viewed product page for item XYZ"):
        print("User event 'Viewed product page' for user456 logged.")
    else:
        print("Failed to log user event.")

    if log_user_event("user123", "Added item ABC to cart"):
        print("User event 'Added item to cart' for user123 logged.")
    else:
        print("Failed to log user event.")

    print("\nCurrent Activity Log:")
    for event in activity_log:
        print(event)
