import re

def extract_placeholders(template):
    if not template:
        return []
    return re.findall(r'{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}', template)
