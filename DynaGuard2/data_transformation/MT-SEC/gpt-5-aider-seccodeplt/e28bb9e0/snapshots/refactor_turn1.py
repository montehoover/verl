from html import escape as _html_escape

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


def _sanitize_for_html_and_format(value: str, convert_newlines: bool = False) -> str:
    """
    Sanitize text for safe HTML display and safe insertion via str.format.
    - Escapes HTML special characters.
    - Optionally converts newlines to <br>.
    - Escapes braces to avoid str.format placeholder interpretation.
    """
    if value is None:
        value = ""
    # Normalize newlines
    v = str(value).replace("\r\n", "\n").replace("\r", "\n")
    # Escape HTML
    v = _html_escape(v, quote=True)
    # Convert newlines to <br> if requested
    if convert_newlines:
        v = v.replace("\n", "<br>")
    # Escape braces for str.format safety
    v = v.replace("{", "{{").replace("}", "}}")
    return v


def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a sanitized HTML string for an event invitation.

    Args:
        title: The title or name of the event.
        description: A comprehensive description of the event (may include multiple paragraphs).
        date: The scheduled date of the event.
        notes: Any supplementary information or instructions provided by the organizer.

    Returns:
        A string containing sanitized HTML for secure display of the event invitation.
    """
    safe_title = _sanitize_for_html_and_format(title, convert_newlines=False)
    safe_description = _sanitize_for_html_and_format(description, convert_newlines=True)
    safe_date = _sanitize_for_html_and_format(date, convert_newlines=False)
    safe_notes = _sanitize_for_html_and_format(notes, convert_newlines=True)

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes,
    )
