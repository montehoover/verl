def create_event_summary(title, date, description):
    """
    Create an HTML string with event details in separate div blocks.
    
    Args:
        title (str): The event title
        date (str): The event date
        description (str): The event description
    
    Returns:
        str: HTML string with formatted event details
    """
    html = f"""<div class="event-summary">
    <div class="event-title">{title}</div>
    <div class="event-date">{date}</div>
    <div class="event-description">{description}</div>
</div>"""
    return html
