from html import escape

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

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

def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Return sanitized HTML for an event invitation using INVITATION_FORMAT.

    Args:
        event_title: The title of the event.
        event_description: A brief description of the event.
        event_date: The date of the event.
        additional_notes: Any additional notes about the event.

    Returns:
        Sanitized HTML string ready for rendering.
    """
    return INVITATION_FORMAT.format(
        event_title=escape(event_title, quote=True),
        event_description=escape(event_description, quote=True),
        event_date=escape(event_date, quote=True),
        additional_notes=escape(additional_notes, quote=True),
    )
