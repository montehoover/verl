# A local dictionary to store valid message identifiers
VALID_MESSAGES = {
    "msg1": "This is the first message.",
    "msg2": "Another valid message.",
    "msg_abc": "Message with id abc.",
}

def is_message_valid(message_id: str, expected_signature: str) -> bool:
    """
    Checks if a given message identifier exists in a local dictionary
    and if its content matches the expected signature.

    Args:
        message_id: The identifier of the message to check.
        expected_signature: The expected content (signature) of the message.

    Returns:
        True if the message_id exists and its content matches the
        expected_signature, False otherwise.
    """
    if message_id in VALID_MESSAGES:
        return VALID_MESSAGES[message_id] == expected_signature
    return False

if __name__ == '__main__':
    # Example Usage
    print(f"Is 'msg1' valid with signature 'This is the first message.'? {is_message_valid('msg1', 'This is the first message.')}")
    print(f"Is 'msg1' valid with signature 'Wrong signature'? {is_message_valid('msg1', 'Wrong signature')}")
    print(f"Is 'msg_xyz' valid with signature 'Any signature'? {is_message_valid('msg_xyz', 'Any signature')}")
    print(f"Is 'msg2' valid with signature 'Another valid message.'? {is_message_valid('msg2', 'Another valid message.')}")
    print(f"Is '' valid with signature ''? {is_message_valid('', '')}")

    VALID_MESSAGES["new_msg"] = "A dynamically added message"
    print(f"Is 'new_msg' valid with signature 'A dynamically added message'? {is_message_valid('new_msg', 'A dynamically added message')}")
    print(f"Is 'new_msg' valid with signature 'Incorrect dynamic signature'? {is_message_valid('new_msg', 'Incorrect dynamic signature')}")
