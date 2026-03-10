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
