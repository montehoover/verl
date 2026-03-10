import re
import logging


# Configure logger
logger = logging.getLogger(__name__)


def validate_email_format(email_address: str) -> bool:
    """Check if the email address matches the expected format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    return bool(re.match(pattern, email_address))


def extract_domain(email_address: str) -> str:
    """Extract the domain portion from a valid email address."""
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    match = re.match(pattern, email_address)
    return match.group(1)


def retrieve_domain_from_email(email_address: str):
    logger.debug(f"Processing email address: {email_address}")
    
    if not email_address:
        logger.warning("Empty email address provided")
        return None
    
    if not validate_email_format(email_address):
        logger.warning(f"Invalid email format: {email_address}")
        return None
    
    domain = extract_domain(email_address)
    logger.info(f"Successfully extracted domain: {domain} from email: {email_address}")
    return domain
