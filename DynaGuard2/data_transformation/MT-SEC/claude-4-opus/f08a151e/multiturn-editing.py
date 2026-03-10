import re

def get_html_tags(html_input: str) -> list:
    # Regular expression pattern to match HTML tags
    tag_pattern = r'<[^>]+>'
    
    try:
        # Find all HTML tags in the input
        tags = re.findall(tag_pattern, html_input)
        return tags
    except Exception:
        # Return empty list if any exception occurs
        return []
