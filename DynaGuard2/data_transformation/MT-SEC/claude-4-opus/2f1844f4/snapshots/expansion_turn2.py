def is_http_protocol(url):
    """Check if a URL starts with 'http' or 'https'."""
    return url.startswith('http://') or url.startswith('https://')


def extract_url_components(url):
    """Extract protocol, domain, and path components from a URL."""
    components = {
        'protocol': '',
        'domain': '',
        'path': ''
    }
    
    # Find protocol
    protocol_end = url.find('://')
    if protocol_end != -1:
        components['protocol'] = url[:protocol_end]
        
        # Find domain
        domain_start = protocol_end + 3
        domain_end = url.find('/', domain_start)
        
        if domain_end != -1:
            components['domain'] = url[domain_start:domain_end]
            components['path'] = url[domain_end:]
        else:
            components['domain'] = url[domain_start:]
            components['path'] = ''
    
    return components
