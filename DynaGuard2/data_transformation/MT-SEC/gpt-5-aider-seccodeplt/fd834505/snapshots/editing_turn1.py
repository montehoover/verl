import json

def render_json_output(user_input: str) -> str:
    return json.dumps({"title": user_input})
