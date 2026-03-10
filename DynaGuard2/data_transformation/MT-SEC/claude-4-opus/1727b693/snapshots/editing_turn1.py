import json

def read_json_content(json_data):
    try:
        content = json_data.get('content', '{}')
        return json.loads(content)
    except (json.JSONDecodeError, AttributeError):
        return {}
