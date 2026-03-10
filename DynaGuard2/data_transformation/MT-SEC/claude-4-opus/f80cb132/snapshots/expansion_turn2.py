from urllib.parse import urlparse

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_url_components(url):
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def is_valid_domain(url):
    parsed = urlparse(url)
    return parsed.netloc in VALID_REDIRECT_DOMAINS
