from html import escape

INVITATION_FORMAT: str = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def generate_event_html(event_title: str, event_date: str) -> str:
    """
    Generate an HTML representation of an event.

    Args:
        event_title (str): The title of the event.
        event_date (str): The date of the event.

    Returns:
        str: HTML string in the format:
             <div><h2>{event_title}</h2><p><strong>Date:</strong> {event_date}</p></div>
    """
    safe_title = escape(str(event_title), quote=True)
    safe_date = escape(str(event_date), quote=True)
    return f"<div><h2>{safe_title}</h2><p><strong>Date:</strong> {safe_date}</p></div>"

def format_notes_html(notes: str) -> str:
    """
    Convert user-provided notes into an HTML-safe string by escaping special characters.

    Args:
        notes (str): The notes provided by the user.

    Returns:
        str: An HTML-safe version of the notes.
    """
    return escape(str(notes), quote=True)

def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Generate a full HTML view of an event invitation.

    Args:
        event_title (str): The title of the event.
        event_description (str): The description of the event.
        event_date (str): The date of the event.
        additional_notes (str): Additional notes provided by the user.

    Returns:
        str: HTML string formatted using INVITATION_FORMAT.
    """
    safe_title = escape(str(event_title), quote=True)
    safe_description = escape(str(event_description), quote=True)
    safe_date = escape(str(event_date), quote=True)
    safe_notes = format_notes_html(additional_notes)

    return INVITATION_FORMAT.format(
        event_title=safe_title,
        event_description=safe_description,
        event_date=safe_date,
        additional_notes=safe_notes,
    )
