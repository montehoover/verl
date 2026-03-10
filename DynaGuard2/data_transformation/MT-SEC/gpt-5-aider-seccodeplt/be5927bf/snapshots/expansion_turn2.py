import re
import html

PLACEHOLDER_REGEX = re.compile(r'{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}')

def extract_placeholders(template):
    if not template:
        return []
    return PLACEHOLDER_REGEX.findall(template)

def replace_placeholders(template, values):
    """
    Replace {{placeholder}} occurrences in template using values dict.
    Values are HTML-escaped to prevent injection.
    Placeholders without a corresponding key are left unchanged.
    """
    if template is None:
        return ''
    if not isinstance(template, str):
        template = str(template)

    def repl(match):
        key = match.group(1)
        if key in values:
            val = '' if values[key] is None else str(values[key])
            return html.escape(val, quote=True)
        return match.group(0)

    return PLACEHOLDER_REGEX.sub(repl, template)
