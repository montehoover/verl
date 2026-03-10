import json

def validate_content_type(json_data):
    try:
        if json_data.get('content_type') == 'application/json':
            content = json_data.get('content', '{}')
            return json.loads(content)
        else:
            return {}
    except (json.JSONDecodeError, AttributeError):
        return {}
