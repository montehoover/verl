import json

def render_json_output(user_input: str) -> str:
    return json.dumps({"title": user_input})

def render_markdown_output(user_input: str) -> str:
    return f"# {user_input}"
