from html import escape

INVITATION_FORMAT = "<div><h2>{event_name}</h2><p>{description_of_event}</p><p><strong>Date:</strong> {date_of_event}</p><p>{notes_for_event}</p></div>"


def generate_event_update(title: str, message: str, date: str) -> str:
    """
    Create a formatted event update string combining title, message, and date.
    """
    title_s = str(title).strip()
    message_s = str(message).strip()
    date_s = str(date).strip()
    return f"{title_s} [{date_s}]: {message_s}"


def create_invitation_for_event(
    event_name: str,
    description_of_event: str,
    date_of_event: str,
    notes_for_event: str
) -> str:
    """
    Create a sanitized HTML invitation string using INVITATION_FORMAT.
    - Escapes HTML special characters to prevent injection.
    - Preserves newlines in description and notes by converting to <br>.
    """
    def _escape_basic(s: str) -> str:
        return escape(str(s).strip(), quote=True)

    def _escape_with_breaks(s: str) -> str:
        text = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
        text = escape(text, quote=True)
        return text.replace("\n", "<br>")

    safe_event_name = _escape_basic(event_name)
    safe_description = _escape_with_breaks(description_of_event)
    safe_date = _escape_basic(date_of_event)
    safe_notes = _escape_with_breaks(notes_for_event)

    return INVITATION_FORMAT.format(
        event_name=safe_event_name,
        description_of_event=safe_description,
        date_of_event=safe_date,
        notes_for_event=safe_notes
    )
