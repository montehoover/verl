import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


def _sanitize_text(text: str, convert_newlines: bool = False) -> str:
    """Sanitizes a string for HTML display and optionally converts newlines."""
    sanitized_text = html.escape(text)
    if convert_newlines:
        sanitized_text = sanitized_text.replace('\n', '<br>')
    return sanitized_text


def _format_invitation_html(title: str, description: str, date: str, notes: str) -> str:
    """Formats the event details into an HTML string."""
    return INVITATION_FORMAT.format(
        title=title,
        description=description,
        date=date,
        notes=notes
    )


def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generates a secure HTML representation of an event invitation.

    Args:
        title: The name or title of the event.
        description: A comprehensive description of the event.
        date: The scheduled date for the event.
        notes: Extra information or instructions.

    Returns:
        A string containing sanitized HTML for the event invitation.
    """
    safe_title = _sanitize_text(title)
    safe_description = _sanitize_text(description, convert_newlines=True)
    safe_date = _sanitize_text(date)
    safe_notes = _sanitize_text(notes, convert_newlines=True)

    return _format_invitation_html(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes
    )
