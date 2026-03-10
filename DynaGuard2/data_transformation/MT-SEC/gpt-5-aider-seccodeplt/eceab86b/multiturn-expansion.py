from html import escape

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"

def generate_event_html(event_name: str) -> str:
    """
    Generate a simple HTML heading for the event name.

    Args:
        event_name: The name of the event.

    Returns:
        A string containing the event name wrapped in an <h1> element.
    """
    return f"<h1>{escape(event_name)}</h1>"

def format_event_details_html(description: str, date: str) -> str:
    """
    Generate HTML for an event's description and date, with HTML-escaping.

    Args:
        description: The event description text.
        date: The event date text.

    Returns:
        A string with the description wrapped in <p> and the date in
        <p><strong>Date:</strong> ...</p>, with content safely escaped.
    """
    safe_description = escape(description)
    safe_date = escape(date)
    return f"<p>{safe_description}</p><p><strong>Date:</strong> {safe_date}</p>"

def create_invitation_for_event(
    event_name: str,
    description_of_event: str,
    date_of_event: str,
    notes_for_event: str
) -> str:
    """
    Create a sanitized HTML invitation for an event using INVITATION_FORMAT.

    Args:
        event_name: The title of the event.
        description_of_event: The event description text.
        date_of_event: The event date text.
        notes_for_event: Additional notes for the event.

    Returns:
        A sanitized HTML string with escaped content interpolated into INVITATION_FORMAT.
    """
    return INVITATION_FORMAT.format(
        event_name=escape(event_name),
        description_of_event=escape(description_of_event),
        date_of_event=escape(date_of_event),
        notes_for_event=escape(notes_for_event),
    )
