chat_rooms = {}

def push_text_to_room(room_name, member_id, content):
    if room_name in chat_rooms:
        if 'messages' not in chat_rooms[room_name]:
            chat_rooms[room_name]['messages'] = {}
        if member_id not in chat_rooms[room_name]['messages']:
            chat_rooms[room_name]['messages'][member_id] = []
        chat_rooms[room_name]['messages'][member_id].append(content)
        return True
    return False
