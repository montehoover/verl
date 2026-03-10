import json

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
