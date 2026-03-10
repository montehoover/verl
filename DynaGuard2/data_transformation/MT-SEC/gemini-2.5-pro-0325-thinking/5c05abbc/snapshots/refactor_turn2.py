# A predefined global dictionary holding the correct cryptographic signatures
# for various API messages. Each key is a message ID, and the value is its
# corresponding signature.
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Confirm the validity of messages received from an external API using their
    cryptographic signature.

    This function checks if a given message ID is known and if the provided
    signature matches the authentic signature stored for that message ID.

    Args:
        msg_id: The ID of the message whose authenticity needs to be verified.
                Expected to be a string.
        sign: The cryptographic signature tied to that specific message.
              Expected to be a string.

    Returns:
        True if the cryptographic signature matches the stored value for the
        given message ID; otherwise, returns False.

    Raises:
        KeyError: If the `msg_id` is not found in the `VALID_API_MESSAGES`
                  dictionary, indicating an unknown or invalid message ID.
    """
    # Check if the provided message ID exists in our record of valid messages.
    if msg_id not in VALID_API_MESSAGES:
        # If the message ID is not found, raise a KeyError to indicate
        # that the message cannot be verified due to an unknown ID.
        raise KeyError(f"Message ID '{msg_id}' not found. Cannot verify signature.")
    
    # Retrieve the expected, authentic signature for the given message ID.
    expected_sign = VALID_API_MESSAGES[msg_id]
    
    # Compare the provided signature with the expected signature.
    return expected_sign == sign
