from urllib.parse import urlparse

def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def check_trusted_domain(url, trusted_domains):
    domain = extract_domain(url)
    return domain in trusted_domains
