def generate_event_html(event_name: str, event_details: str) -> str:
    """
    Generates an HTML string for an event.

    Args:
        event_name: The name of the event.
        event_details: The details of the event.

    Returns:
        An HTML string representing the event.
    """
    return f"<div><h2>{event_name}</h2><p>{event_details}</p></div>"


def format_event_details(description: str, date: str, notes: str) -> str:
    """
    Formats event details for HTML display.

    Args:
        description: The main description of the event.
        date: The date of the event.
        notes: Additional notes for the event.

    Returns:
        An HTML string with formatted event details.
    """
    return f"<p>{description}</p><p><strong>Date: {date}</strong></p><p>Notes: {notes}</p>"
