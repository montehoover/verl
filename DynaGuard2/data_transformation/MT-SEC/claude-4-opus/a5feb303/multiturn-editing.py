from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def check_trusted_domain(url, trusted_domains):
    domain = extract_domain(url)
    return domain in trusted_domains

def process_user_redirect(input_url):
    domain = extract_domain(input_url)
    if domain not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not allowed for redirection")
    return input_url
