VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def verify_api_message(msg_id, sign):
    return msg_id in VALID_API_MESSAGES and VALID_API_MESSAGES[msg_id] == sign
