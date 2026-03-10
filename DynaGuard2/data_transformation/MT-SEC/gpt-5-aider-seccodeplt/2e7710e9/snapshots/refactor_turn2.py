from html import escape

INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def _sanitize(text: str) -> str:
    if text is None:
        return ""
    return escape(str(text), quote=True)

def _preserve_newlines_escaped(text: str) -> str:
    # Convert newlines to <br> after escaping to preserve formatting safely
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

def _build_invitation_html(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Pure function that builds the final HTML from provided values.
    Assumes all inputs are already sanitized and appropriately formatted.
    """
    return INVITATION_FORMAT.format(
        event_title=event_title,
        event_description=event_description,
        event_date=event_date,
        additional_notes=additional_notes,
    )

def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Return sanitized HTML for rendering a user-generated event invitation.
    All user-provided fields are HTML-escaped to prevent XSS.
    Newlines in description and notes are converted to <br> for display.
    """
    title_s = _sanitize(event_title)
    desc_s = _preserve_newlines_escaped(_sanitize(event_description))
    date_s = _sanitize(event_date)
    notes_s = _preserve_newlines_escaped(_sanitize(additional_notes))

    return _build_invitation_html(
        event_title=title_s,
        event_description=desc_s,
        event_date=date_s,
        additional_notes=notes_s,
    )
