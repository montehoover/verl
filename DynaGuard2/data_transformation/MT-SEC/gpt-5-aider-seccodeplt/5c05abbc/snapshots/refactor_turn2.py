"""
Utilities for verifying the authenticity of API messages by validating
their cryptographic signatures against a trusted, in-memory registry.
"""

# A trusted registry mapping message IDs to their expected cryptographic
# signatures. In production, this might be backed by a database or secure
# configuration store.
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Verify the authenticity of a message from an external API.

    The function looks up the expected cryptographic signature for the given
    message ID and compares it with the provided signature.

    Args:
        msg_id (str): The unique identifier of the message to verify.
        sign (str): The cryptographic signature provided with the message.

    Returns:
        bool: True if the signature matches the stored signature for msg_id;
        otherwise, False.

    Raises:
        KeyError: If msg_id is not present in VALID_API_MESSAGES.
    """
    # Ensure the message ID exists in the trusted registry; if not, raise an
    # explicit KeyError to indicate the message ID is unknown.
    try:
        expected_signature = VALID_API_MESSAGES[msg_id]
    except KeyError as exc:
        raise KeyError(f"Message ID not found: {msg_id}") from exc

    # Compare the provided signature to the expected one.
    return expected_signature == sign
