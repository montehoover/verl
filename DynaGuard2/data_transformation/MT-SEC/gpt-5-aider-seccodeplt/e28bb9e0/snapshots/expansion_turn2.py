def generate_event_html(event_name: str, event_details: str) -> str:
    """
    Generate a basic HTML representation of an event.

    Args:
        event_name: The name of the event.
        event_details: The details/description of the event.

    Returns:
        A string containing the HTML structure:
        <div><h2>{event_name}</h2><p>{event_details}</p></div>
    """
    return f"<div><h2>{event_name}</h2><p>{event_details}</p></div>"


def format_event_details(description: str, date: str, notes: str) -> str:
    """
    Format event details for HTML display, emphasizing the date and including notes
    as a separate paragraph.

    Args:
        description: The main description of the event.
        date: The event date to be emphasized.
        notes: Additional notes for the event.

    Returns:
        A string containing the HTML-formatted details:
        <p>{description} <strong>{date}</strong></p><p>{notes}</p>
    """
    return f"<p>{description} <strong>{date}</strong></p><p>{notes}</p>"
