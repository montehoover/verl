# Placeholder for the local dictionary of messages
# This can be populated with actual message data
MESSAGES_DB = {
    "id123": {"text": "Hello world"},
    "id456": {"text": "Another message"},
    "id789": {"text": "Test message"},
}

def check_message_existence(message_id: str) -> bool:
    """
    Checks if a given message ID exists in a local dictionary.

    Args:
        message_id: The ID of the message to check (str).

    Returns:
        True if the message ID is found in MESSAGES_DB, False otherwise.
    """
    return message_id in MESSAGES_DB

if __name__ == '__main__':
    # Example Usage
    print(f"Does 'id123' exist? {check_message_existence('id123')}")
    print(f"Does 'id_unknown' exist? {check_message_existence('id_unknown')}")
    print(f"Does 'id456' exist? {check_message_existence('id456')}")

    # Example of adding a new message and checking
    MESSAGES_DB["new_id_001"] = {"text": "A newly added message"}
    print(f"Does 'new_id_001' exist? {check_message_existence('new_id_001')}")
