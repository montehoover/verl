def basic_event_announcement(event_title: str, event_date: str) -> str:
    """
    Return a simple formatted event announcement string.

    Args:
        event_title: The title of the event.
        event_date: The date of the event.

    Returns:
        A string in the format 'Event: <event_title> on <event_date>'.
    """
    return f"Event: {event_title} on {event_date}"
