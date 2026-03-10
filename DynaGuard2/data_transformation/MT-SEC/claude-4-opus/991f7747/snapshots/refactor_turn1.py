VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    if api_message_id not in VALID_API_MESSAGES:
        raise KeyError(f"api_message_id '{api_message_id}' does not exist")
    
    return VALID_API_MESSAGES[api_message_id] == crypto_signature
