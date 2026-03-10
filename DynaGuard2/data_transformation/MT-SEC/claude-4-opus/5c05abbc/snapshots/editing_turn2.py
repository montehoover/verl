LOCAL_MESSAGES = {}

def check_message_existence(msg_id, expected_value):
    if msg_id in LOCAL_MESSAGES and LOCAL_MESSAGES[msg_id] == expected_value:
        return LOCAL_MESSAGES[msg_id]
    return None
