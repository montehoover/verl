import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def _sanitize_event_data(title: str, description: str, date: str, notes: str) -> dict:
    """Sanitizes event data using HTML escaping."""
    return {
        "title": html.escape(title),
        "description": html.escape(description),
        "date": html.escape(date),
        "notes": html.escape(notes),
    }

def _format_event_html(data: dict, format_string: str) -> str:
    """Formats the event data into an HTML string."""
    return format_string.format(
        title=data["title"],
        description=data["description"],
        date=data["date"],
        notes=data["notes"]
    )

def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.

    Args:
        title: The title or name of the event.
        description: A comprehensive description of the event.
        date: The scheduled date of the event.
        notes: Any supplementary information or instructions.

    Returns:
        A string containing sanitized HTML for the event invitation.
    """
    sanitized_data = _sanitize_event_data(title, description, date, notes)
    return _format_event_html(sanitized_data, INVITATION_FORMAT)
