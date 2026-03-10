from html import escape
from typing import Final

# Predefined HTML template with placeholders for fields
INVITATION_FORMAT: Final[str] = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"


def _sanitize_inline(text: str) -> str:
    """
    Escape HTML-sensitive characters for inline content (single-line).
    """
    return escape(text, quote=True)


def _sanitize_multiline(text: str) -> str:
    """
    Escape HTML-sensitive characters and preserve newlines by converting them to <br>.
    """
    # Normalize newlines to \n
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Escape and then convert newlines to <br> for display within <p> tags
    return escape(normalized, quote=True).replace("\n", "<br>")


def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a sanitized HTML string for an event invitation.

    Args:
        title: The name or title of the event.
        description: A comprehensive description of the event (may contain multiple lines).
        date: The scheduled date for the event.
        notes: Extra information or instructions provided by the user.

    Returns:
        A string containing sanitized HTML for safely displaying the event invitation.
    """
    safe_title = _sanitize_inline(title)
    safe_description = _sanitize_multiline(description)
    safe_date = _sanitize_inline(date)
    safe_notes = _sanitize_multiline(notes)

    return INVITATION_FORMAT.format(
        title=safe_title,
        description=safe_description,
        date=safe_date,
        notes=safe_notes,
    )
