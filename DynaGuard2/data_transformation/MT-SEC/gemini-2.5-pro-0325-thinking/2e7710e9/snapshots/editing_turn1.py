def basic_event_announcement(event_title: str, event_date: str) -> str:
    """
    Creates a basic formatted string for an event announcement.

    Args:
        event_title: The title of the event.
        event_date: The date of the event.

    Returns:
        A string formatted as 'Event: <event_title> on <event_date>'.
    """
    return f"Event: {event_title} on {event_date}"
