import re

def check_url_format(link):
    # Regular expression pattern to validate URL format
    pattern = r'^(https?|ftp|ftps)://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, link))
