chat_rooms = {}

def get_user_messages(room_name, member_id):
    if room_name in chat_rooms and member_id in chat_rooms[room_name]:
        if 'messages' in chat_rooms[room_name] and member_id in chat_rooms[room_name]['messages']:
            return chat_rooms[room_name]['messages'][member_id]
    return []
