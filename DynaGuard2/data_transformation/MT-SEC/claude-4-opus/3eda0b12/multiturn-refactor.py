from urllib import parse as urlparse
import logging

# Configure logger
logger = logging.getLogger(__name__)

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def is_allowed_domain(domain, base_domain, permitted_domains):
    """Check if a domain is either the base domain or in the list of permitted domains."""
    is_allowed = domain == base_domain or domain in permitted_domains
    logger.debug(f"Domain check: '{domain}' - Base: '{base_domain}' - Allowed: {is_allowed}")
    return is_allowed

def validate_url_domain(url, base_domain, permitted_domains):
    """Validate that a URL's domain is allowed."""
    parsed_url = urlparse.urlparse(url)
    logger.debug(f"Validating URL domain: '{parsed_url.netloc}' from URL: '{url}'")
    if not is_allowed_domain(parsed_url.netloc, base_domain, permitted_domains):
        logger.error(f"Domain validation failed: '{parsed_url.netloc}' is not allowed")
        raise ValueError(f"Redirect URL domain '{parsed_url.netloc}' is not allowed")
    logger.debug(f"Domain validation passed for: '{parsed_url.netloc}'")

def normalize_path(path):
    """Ensure a path starts with / for proper URL joining."""
    if not path.startswith('/'):
        normalized = '/' + path
        logger.debug(f"Normalized path: '{path}' -> '{normalized}'")
        return normalized
    logger.debug(f"Path already normalized: '{path}'")
    return path

def construct_full_url(base_url, path):
    """Construct a full URL from a base URL and a path."""
    normalized_path = normalize_path(path)
    result = urlparse.urljoin(base_url, normalized_path)
    logger.debug(f"Constructed full URL: '{result}' from base: '{base_url}' and path: '{normalized_path}'")
    return result

def construct_redirect_url(main_url, target_param):
    logger.info(f"Starting redirect URL construction - Base: '{main_url}', Target: '{target_param}'")
    
    # Parse the main URL to get its components
    main_parsed = urlparse.urlparse(main_url)
    main_domain = main_parsed.netloc
    logger.debug(f"Parsed main URL domain: '{main_domain}'")
    
    # If target_param is empty, return the main URL
    if not target_param:
        logger.info(f"Empty target parameter, returning main URL: '{main_url}'")
        return main_url
    
    # Parse the target parameter
    target_parsed = urlparse.urlparse(target_param)
    logger.debug(f"Parsed target parameter - Scheme: '{target_parsed.scheme}', Netloc: '{target_parsed.netloc}', Path: '{target_parsed.path}'")
    
    # If target_param has a netloc (is a full URL)
    if target_parsed.netloc:
        logger.info(f"Target is a full URL: '{target_param}'")
        # Validate the domain
        validate_url_domain(target_param, main_domain, PERMITTED_DOMAINS)
        # Return the full URL as is
        logger.info(f"Returning validated full URL: '{target_param}'")
        return target_param
    
    # If target_param is a relative path, construct the full URL
    logger.info(f"Target is a relative path: '{target_param}'")
    result = construct_full_url(main_url, target_param)
    
    # Verify the final URL stays within allowed domains
    validate_url_domain(result, main_domain, PERMITTED_DOMAINS)
    
    logger.info(f"Successfully constructed redirect URL: '{result}'")
    return result
