import re

def extract_url_components(url):
    # Regular expression to match protocol and domain
    pattern = r'^(https?|ftp|ftps)://([^/:\s]+(?:\.[^/:\s]+)*)'
    match = re.match(pattern, url)
    
    if match:
        protocol = match.group(1)
        domain = match.group(2)
        return (protocol, domain)
    else:
        # Try to extract domain without protocol
        pattern = r'^([^/:\s]+(?:\.[^/:\s]+)*)'
        match = re.match(pattern, url)
        if match:
            return (None, match.group(1))
        else:
            return (None, None)
