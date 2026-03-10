from urllib.parse import urlparse

def extract_url_components(url):
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'username': parsed.username,
        'password': parsed.password,
        'hostname': parsed.hostname,
        'port': parsed.port
    }

def check_domain_approval(url, allowed_domains):
    parsed = urlparse(url)
    domain = parsed.netloc
    
    is_approved = domain in allowed_domains
    message = 'approved' if is_approved else 'not approved'
    
    return {
        'domain': domain,
        'message': message,
        'is_approved': is_approved
    }
