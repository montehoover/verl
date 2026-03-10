from html import escape

INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    sanitized_title = escape(title, quote=True)
    sanitized_description = escape(description, quote=True)
    sanitized_date = escape(date, quote=True)
    sanitized_notes = escape(notes, quote=True)
    return INVITATION_FORMAT.format(
        title=sanitized_title,
        description=sanitized_description,
        date=sanitized_date,
        notes=sanitized_notes,
    )
