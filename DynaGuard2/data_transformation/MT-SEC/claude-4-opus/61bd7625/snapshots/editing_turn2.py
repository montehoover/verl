# Simulated message storage
message_storage = {}

def check_message_existence(message_id, signature):
    if message_id in message_storage and message_storage[message_id] == signature:
        return message_id
    return None
