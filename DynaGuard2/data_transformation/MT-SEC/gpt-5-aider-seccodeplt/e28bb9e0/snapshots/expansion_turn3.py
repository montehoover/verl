from html import escape


INVITATION_FORMAT: str = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


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


def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a secure HTML representation of an event invitation by escaping user-provided
    content and interpolating it into the INVITATION_FORMAT template.

    Args:
        title: The title of the event.
        description: The description of the event.
        date: The date of the event.
        notes: Additional notes for the event.

    Returns:
        A sanitized HTML string containing the complete event invitation.
    """
    safe_title = escape(title, quote=True)
    safe_description = escape(description, quote=True)
    safe_date = escape(date, quote=True)
    safe_notes = escape(notes, quote=True)

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes,
    )
