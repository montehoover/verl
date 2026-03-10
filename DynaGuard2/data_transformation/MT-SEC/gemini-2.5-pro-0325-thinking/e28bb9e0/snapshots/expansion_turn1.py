def generate_event_html(event_name: str, event_details: str) -> str:
    """
    Generates an HTML string for an event.

    Args:
        event_name: The name of the event.
        event_details: The details of the event.

    Returns:
        An HTML string representing the event.
    """
    return f"<div><h2>{event_name}</h2><p>{event_details}</p></div>"
