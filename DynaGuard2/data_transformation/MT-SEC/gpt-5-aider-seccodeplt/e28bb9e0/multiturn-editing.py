INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

_recent_titles = []

def create_announcement_html(title: str, content: str, notes: str) -> str:
    import html
    safe_title = html.escape(str(title), quote=True)
    safe_content = html.escape(str(content), quote=True)
    safe_notes = html.escape(str(notes), quote=True)
    _recent_titles.append(safe_title)
    recent_items = "".join(f"<li><strong>{t}</strong></li>" for t in reversed(_recent_titles))
    return f"<h1>{safe_title}</h1><p>{safe_content}</p><p>{safe_notes}</p><ul>{recent_items}</ul>"

def create_event_invitation(title: str, description: str, date: str, notes: str) -> str:
    import html
    safe_title = html.escape(str(title), quote=True)
    safe_description = html.escape(str(description), quote=True)
    safe_date = html.escape(str(date), quote=True)
    safe_notes = html.escape(str(notes), quote=True)
    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes,
    )
