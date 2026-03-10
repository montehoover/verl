def detailed_event_announcement(event_title: str, event_description: str, event_date: str) -> str:
    """
    Creates a detailed formatted string for an event announcement.

    Args:
        event_title: The title of the event.
        event_description: The description of the event.
        event_date: The date of the event.

    Returns:
        A string formatted as 'Event: <event_title>, Description: <event_description>, Date: <event_date>'.
    """
    return f"Event: {event_title}, Description: {event_description}, Date: {event_date}"
