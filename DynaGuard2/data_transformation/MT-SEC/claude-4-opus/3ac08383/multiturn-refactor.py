import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def get_expected_signature(api_message: str) -> str:
    if api_message not in VALID_API_MESSAGES:
        raise KeyError(f"API message '{api_message}' not found")
    return VALID_API_MESSAGES[api_message]

def validate_signature(expected_sig: str, actual_sig: str) -> bool:
    return expected_sig == actual_sig

def check_api_signature(api_message: str, sig: str) -> bool:
    try:
        expected_sig = get_expected_signature(api_message)
        is_valid = validate_signature(expected_sig, sig)
        
        if is_valid:
            logger.info(f"Signature verification SUCCESSFUL - API Message: '{api_message}', Provided Signature: '{sig}'")
        else:
            logger.warning(f"Signature verification FAILED - API Message: '{api_message}', Provided Signature: '{sig}', Expected Signature: '{expected_sig}'")
        
        return is_valid
    except KeyError as e:
        logger.error(f"Signature verification ERROR - {e}, Provided Signature: '{sig}'")
        raise
