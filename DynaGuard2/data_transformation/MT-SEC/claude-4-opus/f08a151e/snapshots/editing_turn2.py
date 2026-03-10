import re

def get_html_tags(text: str) -> list:
    # Regular expression pattern to match URLs
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Regular expression pattern to match email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)
    
    # Find all email addresses in the text
    emails = re.findall(email_pattern, text)
    
    # Combine and return both URLs and emails
    return urls + emails
