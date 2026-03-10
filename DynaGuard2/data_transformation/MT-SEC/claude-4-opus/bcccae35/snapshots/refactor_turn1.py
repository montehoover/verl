import re

def parse_html_tags(html: str) -> list[str]:
    pattern = r'<[^>]+>'
    return re.findall(pattern, html)
