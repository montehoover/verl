# A global dictionary to store valid API message identifiers and their expected cryptographic signatures
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def check_api_signature(api_message: str, sig: str) -> bool:
    """
    Verifies the legitimacy of a message from an external API using its cryptographic signature.

    Args:
        api_message: The identifier for the message.
        sig: The cryptographic signature to verify.

    Returns:
        True if the signature is valid for the given message, False otherwise.

    Raises:
        KeyError: If the api_message does not exist in VALID_API_MESSAGES.
    """
    if api_message not in VALID_API_MESSAGES:
        raise KeyError(f"API message '{api_message}' not found.")
    return VALID_API_MESSAGES[api_message] == sig

if __name__ == '__main__':
    # Example Usage
    # Test case 1: Valid message and signature
    print(f"Signature check for 'msg_1001' (valid): {check_api_signature('msg_1001', 'abcd1234efgh5678ijkl9012mnop3456')}")

    # Test case 2: Valid message, invalid signature
    print(f"Signature check for 'msg_1002' (invalid sig): {check_api_signature('msg_1002', 'invalid_signature_string')}")

    # Test case 3: Non-existent message (should raise KeyError)
    try:
        check_api_signature('msg_non_existent', 'any_signature')
    except KeyError as e:
        print(f"Signature check for 'msg_non_existent' (KeyError): {e}")

    # Test case 4: Add a new message and test
    VALID_API_MESSAGES["msg_1003"] = "newkeysig123"
    print(f"Signature check for 'msg_1003' (valid after add): {check_api_signature('msg_1003', 'newkeysig123')}")
    print(f"Signature check for 'msg_1003' (invalid sig after add): {check_api_signature('msg_1003', 'wrongnewsig')}")
