import html

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"


def _escape_and_preserve_newlines(value: str) -> str:
    """
    Escape HTML special characters and convert newlines to <br> for safe display.
    """
    if value is None:
        value = ""
    # Normalize newlines to \n
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    escaped = html.escape(value, quote=True)
    return escaped.replace("\n", "<br>")


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
        description_of_event: Comprehensive description (may contain multiple lines).
        date_of_event: The scheduled date for the event.
        notes_for_event: Extra information or instructions.

    Returns:
        A string containing sanitized HTML for safely displaying the event invitation.
    """
    safe_event_name = _escape_and_preserve_newlines(event_name)
    safe_description = _escape_and_preserve_newlines(description_of_event)
    safe_date = _escape_and_preserve_newlines(date_of_event)
    safe_notes = _escape_and_preserve_newlines(notes_for_event)

    return INVITATION_FORMAT.format(
        event_name=safe_event_name,
        description_of_event=safe_description,
        date_of_event=safe_date,
        notes_for_event=safe_notes,
    )
