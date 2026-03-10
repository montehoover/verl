# List to store recent titles
_recent_titles = []

def create_announcement_html(title: str, content: str, notes: str) -> str:
    """
    Generates an HTML string for an announcement, including title, content, notes,
    and a list of recent announcement titles.

    Args:
        title: The title of the announcement (will be an <h1>).
        content: The main content of the announcement (will be a <p>).
        notes: Additional notes for the announcement (will be a <p>).

    Returns:
        An HTML formatted string.
    """
    global _recent_titles
    # Add current title to the list of recent titles
    # To keep the list manageable, let's say we store the last 5 titles.
    # You can adjust this number as needed.
    _recent_titles.append(title)
    if len(_recent_titles) > 5:
        _recent_titles.pop(0)

    html_elements = [
        f"<h1>{title}</h1>",
        f"<p>{content}</p>"
    ]

    if notes:
        html_elements.append(f"<p><em>Notes: {notes}</em></p>")

    if _recent_titles:
        recent_titles_html = "<h2>Recent Titles:</h2>\n<ul>"
        # Display in reverse chronological order (most recent first)
        for rt_title in reversed(_recent_titles):
            recent_titles_html += f"<li><b>{rt_title}</b></li>"
        recent_titles_html += "</ul>"
        html_elements.append(recent_titles_html)

    return "\n".join(html_elements)
