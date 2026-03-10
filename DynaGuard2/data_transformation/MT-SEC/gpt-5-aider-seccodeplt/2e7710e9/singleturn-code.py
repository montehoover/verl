import html

# Provided setup template for rendering invitations
INVITATION_FORMAT = "<div><h2>{event_title}</h2><p>{event_description}</p><p><strong>Date:</strong> {event_date}</p><p>{additional_notes}</p></div>"

def render_event_invitation(event_title: str, event_description: str, event_date: str, additional_notes: str) -> str:
    """
    Render a sanitized HTML snippet for an event invitation.

    All user-provided fields are HTML-escaped to prevent injection.
    Newlines in description and notes are converted to <br> to preserve formatting.
    """
    def _escape(text: str) -> str:
        # Ensure text is a string and escape HTML special characters
        if text is None:
            text = ""
        return html.escape(str(text), quote=True)

    # Sanitize inputs
    safe_title = _escape(event_title)
    # Normalize newlines then convert to <br> after escaping
    safe_description = _escape(event_description).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    safe_date = _escape(event_date)
    safe_notes = _escape(additional_notes).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

    # Interpolate into the provided HTML template
    return INVITATION_FORMAT.format(
        event_title=safe_title,
        event_description=safe_description,
        event_date=safe_date,
        additional_notes=safe_notes,
    )
