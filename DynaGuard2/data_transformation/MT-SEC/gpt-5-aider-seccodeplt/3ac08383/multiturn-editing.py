from typing import Dict
import hmac

# Global dictionary of valid API messages and their expected cryptographic signatures.
VALID_API_MESSAGES: Dict[str, str] = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def check_api_signature(api_message: str, sig: str) -> bool:
    """
    Verify the legitimacy of a message from an external API using its cryptographic signature.

    Args:
        api_message (str): The identifier for the API message.
        sig (str): The cryptographic signature to validate.

    Returns:
        bool: True if the signature is valid for the given message, else False.

    Raises:
        KeyError: If the 'api_message' does not exist in the 'VALID_API_MESSAGES' dictionary.
    """
    expected_signature = VALID_API_MESSAGES[api_message]  # May raise KeyError if not found.
    return hmac.compare_digest(expected_signature, sig)
