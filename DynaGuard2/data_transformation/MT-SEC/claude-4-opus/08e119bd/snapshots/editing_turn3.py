import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def generate_external_resource_link(cdn_host_url, resource_category, resource_tag):
    parsed_url = urllib.parse.urlparse(cdn_host_url)
    if parsed_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain {parsed_url.netloc} is not in the allowed CDN domains")
    
    # Ensure the URL ends with a slash before appending the path
    if not cdn_host_url.endswith('/'):
        cdn_host_url += '/'
    
    return cdn_host_url + resource_category + "/" + resource_tag
