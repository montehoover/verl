# A local dictionary to store valid message identifiers
VALID_MESSAGES = {
    "msg1": "This is the first message.",
    "msg2": "Another valid message.",
    "msg_abc": "Message with id abc.",
}

def is_message_valid(message_id: str) -> bool:
    """
    Checks if a given message identifier exists in a local dictionary.

    Args:
        message_id: The identifier of the message to check.

    Returns:
        True if the message_id exists in the VALID_MESSAGES dictionary, 
        False otherwise.
    """
    return message_id in VALID_MESSAGES

if __name__ == '__main__':
    # Example Usage
    print(f"Is 'msg1' valid? {is_message_valid('msg1')}")
    print(f"Is 'msg_xyz' valid? {is_message_valid('msg_xyz')}")
    print(f"Is 'msg2' valid? {is_message_valid('msg2')}")
    print(f"Is '' valid? {is_message_valid('')}")

    VALID_MESSAGES["new_msg"] = "A dynamically added message"
    print(f"Is 'new_msg' valid? {is_message_valid('new_msg')}")
