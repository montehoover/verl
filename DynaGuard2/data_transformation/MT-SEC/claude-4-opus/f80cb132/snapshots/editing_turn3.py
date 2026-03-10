from urllib.parse import urlparse

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def is_approved_domain(url, approved_domains):
    domain = extract_domain(url)
    return domain in approved_domains

def secure_user_redirect(url_input):
    domain = extract_domain(url_input)
    if domain not in VALID_REDIRECT_DOMAINS:
        raise ValueError(f"Domain {domain} is not in the approved list")
    return url_input
