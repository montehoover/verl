from typing import List, Dict, Any

chat_rooms = {}

def get_user_messages(room_name: str, member_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all messages sent by a specific user within a chat room.

    Args:
        room_name: The identifier of the chat room.
        member_id: The unique identifier of the user.

    Returns:
        A list of messages (dictionaries) sent by the user in the specified chat room.
        Returns an empty list if the room does not exist, the user is not a member,
        or the user has sent no messages.
    """
    if room_name not in chat_rooms:
        return []

    room_data = chat_rooms[room_name]

    # Expected structure for room_data:
    # {
    #   "members": {"member_id_1", "member_id_2"},  # A set of member IDs
    #   "messages": [                               # A list of message dictionaries
    #     {"sender_id": "member_id_1", "text": "Hello!"},
    #     # ... other messages
    #   ]
    # }
    if not isinstance(room_data, dict) or \
       "members" not in room_data or \
       not isinstance(room_data.get("members"), set) or \
       "messages" not in room_data or \
       not isinstance(room_data.get("messages"), list):
        # Malformed room_data or room_data does not follow the expected structure.
        # This could also log a warning or raise an error in a real application.
        return []

    # Check if the user is a member of the room
    if member_id not in room_data["members"]:
        return []

    # Filter messages sent by the user
    user_messages: List[Dict[str, Any]] = []
    for message in room_data["messages"]:
        # Ensure message is a dictionary and has 'sender_id'
        if isinstance(message, dict) and message.get("sender_id") == member_id:
            user_messages.append(message)

    return user_messages
