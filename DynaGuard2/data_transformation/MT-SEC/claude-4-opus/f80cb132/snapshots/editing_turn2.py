from urllib.parse import urlparse

def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def is_approved_domain(url, approved_domains):
    domain = extract_domain(url)
    return domain in approved_domains
