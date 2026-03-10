def detailed_event_announcement(event_title: str, event_description: str, event_date: str) -> str:
    """
    Return a formatted event announcement string with title, description, and date.

    Args:
        event_title: The title of the event.
        event_description: A brief description of the event.
        event_date: The date of the event.

    Returns:
        A string in the format 'Event: <event_title>, Description: <event_description>, Date: <event_date>'.
    """
    return f"Event: {event_title}, Description: {event_description}, Date: {event_date}"
