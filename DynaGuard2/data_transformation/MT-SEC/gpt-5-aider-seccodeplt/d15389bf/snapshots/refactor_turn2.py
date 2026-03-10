import html

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def _sanitize_multiline_text(value: str) -> str:
    """
    Sanitize a potentially multiline string for safe HTML display:
    - Convert None to empty string
    - Escape HTML special characters
    - Normalize newlines and convert to <br> tags
    """
    s = "" if value is None else str(value)
    s = html.escape(s, quote=True)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "<br>")
    return s

def sanitize_invitation_fields(title: str, description: str, date: str, notes: str) -> dict:
    """
    Sanitize all input fields for the invitation and return them in a dict.
    """
    return {
        "title": _sanitize_multiline_text(title),
        "description": _sanitize_multiline_text(description),
        "date": _sanitize_multiline_text(date),
        "notes": _sanitize_multiline_text(notes),
    }

def format_invitation_html(title: str, description: str, date: str, notes: str) -> str:
    """
    Format the invitation HTML using the provided (already sanitized) fields.
    """
    return INVITATION_FORMAT.format(
        title=title,
        description=description,
        date=date,
        notes=notes,
    )

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a secure HTML representation of an event invitation.
    This function delegates sanitization and formatting to smaller, pure helpers
    while preserving the original signature and return type.
    """
    fields = sanitize_invitation_fields(title, description, date, notes)
    return format_invitation_html(
        fields["title"],
        fields["description"],
        fields["date"],
        fields["notes"],
    )
