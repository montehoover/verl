"""
Utilities for generating secure HTML event invitations.

This module provides small, pure helper functions to:
- Sanitize user-provided text so it is safe to embed in HTML.
- Format sanitized fields into a predefined HTML template.

Logging:
- Uses a module-level logger to record key information (event title and date)
  whenever an invitation is generated.
- By default, a NullHandler is attached so importing this module does not
  configure logging globally. Applications should configure logging as desired.
"""

import html
import logging

# Predefined HTML template with named placeholders.
INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

# Module-level logger. Applications can configure handlers/levels on demand.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _sanitize_multiline_text(value: str) -> str:
    """
    Sanitize a potentially multiline string for safe HTML display.

    Steps performed:
    - Convert None to an empty string.
    - Escape HTML special characters (&, <, >, ", ').
    - Normalize all newline variations to '\n' and then convert to <br> tags.

    Args:
        value: The input string to sanitize. May be None.

    Returns:
        A safe HTML string with special characters escaped and newlines rendered as <br>.
    """
    # Normalize None to an empty string to simplify downstream processing.
    s = "" if value is None else str(value)

    # Escape HTML special characters to prevent injection/XSS.
    s = html.escape(s, quote=True)

    # Normalize different newline styles to '\n' to ensure consistent rendering.
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Convert newlines to <br> to preserve line breaks in HTML.
    s = s.replace("\n", "<br>")

    return s


def sanitize_invitation_fields(title: str, description: str, date: str, notes: str) -> dict:
    """
    Sanitize all input fields for the invitation and return them in a dict.

    Each field is processed to be safe for direct insertion into HTML.
    Newlines are preserved using <br> tags.

    Args:
        title: Event title.
        description: Detailed event description (may contain multiple lines).
        date: Event date string.
        notes: Additional notes or instructions.

    Returns:
        A dictionary with the same keys ('title', 'description', 'date', 'notes')
        containing sanitized strings safe for HTML output.
    """
    # Apply the same sanitization consistently to all user-supplied fields.
    return {
        "title": _sanitize_multiline_text(title),
        "description": _sanitize_multiline_text(description),
        "date": _sanitize_multiline_text(date),
        "notes": _sanitize_multiline_text(notes),
    }


def format_invitation_html(title: str, description: str, date: str, notes: str) -> str:
    """
    Format the invitation HTML using the provided fields.

    Note:
        This function expects the inputs to be pre-sanitized for HTML.
        It performs no escaping itself.

    Args:
        title: Sanitized event title.
        description: Sanitized event description.
        date: Sanitized event date string.
        notes: Sanitized additional notes.

    Returns:
        The finalized HTML string for the invitation.
    """
    # Interpolate sanitized values into the predefined HTML template.
    return INVITATION_FORMAT.format(
        title=title,
        description=description,
        date=date,
        notes=notes,
    )


def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a secure HTML representation of an event invitation.

    This function delegates:
      - Sanitization to `sanitize_invitation_fields`.
      - HTML assembly to `format_invitation_html`.

    Logging:
      - Logs an INFO message with the event title and date to aid observability.
        Sensitive free-form content (description/notes) is not logged.

    Args:
        title: The name or title of the event.
        description: A comprehensive description of the event (may be multiline).
        date: The scheduled date for the event.
        notes: Extra information or instructions.

    Returns:
        A string containing sanitized HTML for safely displaying the event invitation.
    """
    # Log key information for traceability. Avoid logging verbose/sensitive fields.
    logger.info("Generating event invitation: title=%r, date=%r", title, date)

    # Sanitize the input fields to ensure safe HTML content.
    fields = sanitize_invitation_fields(title, description, date, notes)

    # Format the final invitation HTML using the sanitized fields.
    html_output = format_invitation_html(
        fields["title"],
        fields["description"],
        fields["date"],
        fields["notes"],
    )

    # Optional low-level detail for debugging (kept at DEBUG level).
    logger.debug("Invitation HTML generated (length=%d characters).", len(html_output))

    return html_output
