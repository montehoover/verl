VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Confirms the validity of messages received from an external API using their cryptographic signature.

    Args:
        api_message_id: The ID of the message whose authenticity needs to be verified (str).
        cryptographic_sign: The cryptographic signature to verify (str).

    Returns:
        True if the cryptographic signature matches the stored value for the given message ID,
        False otherwise.

    Raises:
        KeyError: If the api_message_id is not found in the VALID_API_MESSAGES dictionary.
    """
    if api_message_id not in VALID_API_MESSAGES:
        raise KeyError(f"Message ID '{api_message_id}' not found in VALID_API_MESSAGES.")
    
    return VALID_API_MESSAGES[api_message_id] == cryptographic_sign

if __name__ == '__main__':
    # Example Usage
    # Test case 1: Valid message ID and correct signature
    print(f"Validation for 'msg_1001' with correct signature: {validate_api_message('msg_1001', 'abcd1234efgh5678ijkl9012mnop3456')}")

    # Test case 2: Valid message ID and incorrect signature
    print(f"Validation for 'msg_1002' with incorrect signature: {validate_api_message('msg_1002', 'wrong_signature_here_xxxx')}")

    # Test case 3: Non-existent message ID (should raise KeyError)
    try:
        print(f"Validation for 'msg_9999' (non-existent): {validate_api_message('msg_9999', 'any_signature')}")
    except KeyError as e:
        print(f"Error for 'msg_9999': {e}")

    # Test case 4: Another valid message ID and correct signature
    VALID_API_MESSAGES["msg_1003"] = "zyxw9876vuts5432rqpo1098nmlk7654" # Add a new one for testing
    print(f"Validation for 'msg_1003' with correct signature: {validate_api_message('msg_1003', 'zyxw9876vuts5432rqpo1098nmlk7654')}")
