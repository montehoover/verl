import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def create_resource_url(base_cdn_url, res_id, res_type):
    # Construct the complete URL
    url = urllib.parse.urljoin(base_cdn_url, f"{res_type}/{res_id}")
    
    # Parse the URL to extract the domain
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"URL domain '{domain}' is not in the allowed CDN domains")
    
    return url
