import json
import html

def render_json_output(user_input: str) -> str:
    """
    Generates a simple JSON response.

    Args:
        user_input: The string to be used as the title.

    Returns:
        A JSON string like {"title": "My Page"}.
    """
    data = {"title": user_input}
    return json.dumps(data)

def render_markdown_output(user_input: str) -> str:
    """
    Embeds user content into a Markdown level-1 heading.

    Args:
        user_input: The string to be inserted into the Markdown heading.

    Returns:
        A Markdown formatted string like "# {user_input}".
    """
    return f"# {user_input}"

def render_html_output(user_input: str) -> str:
    """
    Embeds a user-provided string into a predefined HTML template safely.

    Args:
        user_input: The string to be inserted into the HTML template.

    Returns:
        The resulting HTML page with the safe user content.
    """
    html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"
    safe_user_content = html.escape(user_input)
    return html_template.format(user_content=safe_user_content)
