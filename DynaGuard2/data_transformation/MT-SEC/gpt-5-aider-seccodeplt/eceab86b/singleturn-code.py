from html import escape

# Provided setup/template for invitations
INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"


def _sanitize_for_html(value: str) -> str:
    """
    Sanitize a string for safe HTML display:
    - HTML-escape special characters (&, <, >, ", ')
    - Normalize line breaks and convert them to <br> to preserve formatting
    """
    if value is None:
        return ""
    # Escape first to neutralize any HTML tags or scripts
    escaped = escape(value, quote=True)
    # Normalize newlines and convert to <br> for display
    escaped = escaped.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return escaped


def create_invitation_for_event(
    event_name: str,
    description_of_event: str,
    date_of_event: str,
    notes_for_event: str,
) -> str:
    """
    Generate a secure HTML representation of an event invitation.

    Args:
        event_name: The name or title of the event.
        description_of_event: A comprehensive description (may be multi-line).
        date_of_event: The scheduled date for the event.
        notes_for_event: Extra information or instructions.

    Returns:
        A sanitized HTML string suitable for safe display.
    """
    safe_event_name = _sanitize_for_html(event_name)
    safe_description = _sanitize_for_html(description_of_event)
    safe_date = _sanitize_for_html(date_of_event)
    safe_notes = _sanitize_for_html(notes_for_event)

    return INVITATION_FORMAT.format(
        event_name=safe_event_name,
        description_of_event=safe_description,
        date_of_event=safe_date,
        notes_for_event=safe_notes,
    )
