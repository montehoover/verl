import json
import html

def render_json_output(user_input: str) -> str:
    return json.dumps({"title": user_input})

def render_markdown_output(user_input: str) -> str:
    return f"# {user_input}"

def render_html_output(user_input: str) -> str:
    html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"
    safe_content = html.escape(user_input)
    return html_template.format(user_content=safe_content)
