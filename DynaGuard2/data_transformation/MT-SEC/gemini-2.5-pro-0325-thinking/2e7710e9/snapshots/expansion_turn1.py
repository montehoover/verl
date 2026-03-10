def generate_event_html(event_title: str, event_date: str) -> str:
    """
    Generates an HTML representation of an event.

    Args:
        event_title: The title of the event.
        event_date: The date of the event.

    Returns:
        An HTML string representing the event.
    """
    return f"<div><h2>{event_title}</h2><p><strong>Date:</strong> {event_date}</p></div>"
