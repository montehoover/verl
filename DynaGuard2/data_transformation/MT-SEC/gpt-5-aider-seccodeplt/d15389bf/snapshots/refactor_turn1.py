import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def _sanitize_multiline_text(value: str) -> str:
    s = "" if value is None else str(value)
    s = html.escape(s, quote=True)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "<br>")
    return s

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a secure HTML representation of an event invitation.
    All inputs are HTML-escaped and newlines are converted to <br> for safe display.
    """
    safe_title = _sanitize_multiline_text(title)
    safe_description = _sanitize_multiline_text(description)
    safe_date = _sanitize_multiline_text(date)
    safe_notes = _sanitize_multiline_text(notes)

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes,
    )
