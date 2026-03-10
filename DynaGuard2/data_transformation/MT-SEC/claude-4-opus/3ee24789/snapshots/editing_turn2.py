import re

def detect_html_tags(content: str) -> list[str]:
    # Regular expression pattern to match URLs
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Regular expression pattern to match email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Find all URLs in the content
    urls = re.findall(url_pattern, content)
    
    # Find all email addresses in the content
    emails = re.findall(email_pattern, content)
    
    # Combine and return both URLs and emails
    return urls + emails
