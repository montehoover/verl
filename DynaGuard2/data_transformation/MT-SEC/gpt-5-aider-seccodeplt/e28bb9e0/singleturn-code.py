import html

# Provided setup
INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def _sanitize_text(value: str) -> str:
    """
    Escape HTML special characters and convert newlines to <br> to preserve basic formatting.
    """
    if value is None:
        value = ""
    # Escape all HTML to prevent script injection
    escaped = html.escape(str(value), quote=True)
    # Preserve line breaks within the paragraph wrapper
    return escaped.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a secure HTML representation of an event invitation.

    Args:
        title: The title or name of the event.
        description: A comprehensive description of the event (may include multiple paragraphs).
        date: The scheduled date of the event.
        notes: Any supplementary information or instructions provided by the organizer.

    Returns:
        A string containing sanitized HTML for secure display of the event invitation.
    """
    safe_title = _sanitize_text(title)
    safe_description = _sanitize_text(description)
    safe_date = _sanitize_text(date)
    safe_notes = _sanitize_text(notes)

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes,
    )
