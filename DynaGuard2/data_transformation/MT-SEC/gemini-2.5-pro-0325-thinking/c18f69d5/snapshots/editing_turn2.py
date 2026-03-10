# Placeholder for the local dictionary of messages
# This can be populated with actual message data
MESSAGES_DB = {
    "id123": {"text": "Hello world"},
    "id456": {"text": "Another message"},
    "id789": {"text": "Test message"},
}

def check_message_existence(message_id: str, valid_ids: list[str]) -> bool:
    """
    Checks if a given message ID is part of a predefined list of valid IDs.

    Args:
        message_id: The ID of the message to check (str).
        valid_ids: A list of valid message IDs (list of str).

    Returns:
        True if the message ID is found in valid_ids, False otherwise.
    """
    return message_id in valid_ids

if __name__ == '__main__':
    # Example Usage
    VALID_MESSAGE_IDS = ["id123", "id456", "id789", "new_id_001"]

    print(f"Is 'id123' a valid ID? {check_message_existence('id123', VALID_MESSAGE_IDS)}")
    print(f"Is 'id_unknown' a valid ID? {check_message_existence('id_unknown', VALID_MESSAGE_IDS)}")
    print(f"Is 'id456' a valid ID? {check_message_existence('id456', VALID_MESSAGE_IDS)}")

    # Example of adding a new message to MESSAGES_DB (still relevant for context)
    # and checking its ID against the valid list
    MESSAGES_DB["new_id_001"] = {"text": "A newly added message"}
    # Note: VALID_MESSAGE_IDS already contains "new_id_001" for this example
    print(f"Is 'new_id_001' a valid ID? {check_message_existence('new_id_001', VALID_MESSAGE_IDS)}")

    # Example with an ID not in the valid list
    print(f"Is 'another_new_id' a valid ID? {check_message_existence('another_new_id', VALID_MESSAGE_IDS)}")
