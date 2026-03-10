import logging
from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

# Configure logger
logger = logging.getLogger(__name__)

def is_domain_allowed(domain, app_domain):
    """Check if a domain is either the app domain or in the approved list."""
    is_allowed = domain == app_domain or domain in APPROVED_DOMAINS
    logger.debug(f"Domain validation: '{domain}' - Allowed: {is_allowed} (App domain: '{app_domain}')")
    return is_allowed

def validate_url_domain(url, app_domain):
    """Validate that a URL's domain is allowed."""
    parsed = urlparse.urlparse(url)
    logger.debug(f"Validating URL domain: '{parsed.netloc}' from URL: '{url}'")
    
    if not is_domain_allowed(parsed.netloc, app_domain):
        logger.error(f"Domain validation failed: '{parsed.netloc}' is not in allowed domains")
        raise ValueError(f"Redirect URL domain '{parsed.netloc}' is not allowed")
    
    logger.debug(f"Domain validation passed for: '{parsed.netloc}'")

def construct_redirect_url(app_url, redirect_param):
    """Construct the final redirect URL from app URL and redirect parameter."""
    redirect_parsed = urlparse.urlparse(redirect_param)
    
    if redirect_parsed.netloc:
        # It's a full URL
        logger.debug(f"Redirect parameter is a full URL: '{redirect_param}'")
        return redirect_param
    else:
        # It's a path - join it with the app URL
        final_url = urlparse.urljoin(app_url, redirect_param)
        logger.debug(f"Redirect parameter is a path. Joined '{redirect_param}' with '{app_url}' to get: '{final_url}'")
        return final_url

def generate_redirect_link(app_url, redirect_param):
    logger.info(f"Generating redirect link - App URL: '{app_url}', Redirect param: '{redirect_param}'")
    
    # Parse the base application URL
    app_parsed = urlparse.urlparse(app_url)
    app_domain = app_parsed.netloc
    logger.debug(f"Parsed app domain: '{app_domain}'")
    
    # Construct the final redirect URL
    final_url = construct_redirect_url(app_url, redirect_param)
    logger.debug(f"Constructed redirect URL: '{final_url}'")
    
    # Validate the final URL's domain
    validate_url_domain(final_url, app_domain)
    
    logger.info(f"Successfully generated redirect URL: '{final_url}'")
    return final_url
