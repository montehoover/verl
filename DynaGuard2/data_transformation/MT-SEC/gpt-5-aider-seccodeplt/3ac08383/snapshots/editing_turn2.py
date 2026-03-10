from typing import Dict

# Local dictionary of messages: keys are message identifiers (str).
# Values are simple string signatures associated with each message.
MESSAGES: Dict[str, str] = {}


def is_message_valid(message_id: str, expected_signature: str) -> bool:
    """
    Check whether the given message identifier exists in the local MESSAGES dictionary
    and that its associated signature matches the expected signature.

    Args:
        message_id (str): The message identifier to look up.
        expected_signature (str): The expected signature string to validate against.

    Returns:
        bool: True if the message exists and the signature matches, False otherwise.
    """
    actual_signature = MESSAGES.get(message_id)
    return actual_signature == expected_signature
