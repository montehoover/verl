from html import escape

def generate_event_html(event_title: str, event_date: str) -> str:
    """
    Generate an HTML representation of an event.

    Args:
        event_title (str): The title of the event.
        event_date (str): The date of the event.

    Returns:
        str: HTML string in the format:
             <div><h2>{event_title}</h2><p><strong>Date:</strong> {event_date}</p></div>
    """
    safe_title = escape(str(event_title), quote=True)
    safe_date = escape(str(event_date), quote=True)
    return f"<div><h2>{safe_title}</h2><p><strong>Date:</strong> {safe_date}</p></div>"
