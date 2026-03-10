from typing import List, Dict, Any

chat_rooms = {}

def push_text_to_room(room_name: str, member_id: str, content: str) -> bool:
    """
    Adds a new message to the specified chat room's history.

    Args:
        room_name: The identifier of the chat room.
        member_id: The unique identifier of the user sending the message.
        content: The text content of the message.

    Returns:
        True if the message was successfully added, False otherwise.
        Reasons for failure include: room does not exist, user is not a member
        of the room, or message content/length limits are exceeded (not yet implemented).
    """
    if room_name not in chat_rooms:
        return False  # Room does not exist

    room_data = chat_rooms[room_name]

    # Validate room_data structure (similar to previous function)
    # Expected structure for room_data:
    # {
    #   "members": {"member_id_1", "member_id_2"},  # A set of member IDs
    #   "messages": [                               # A list of message dictionaries
    #     {"sender_id": "member_id_1", "text": "Hello!"},
    #     # ... other messages
    #   ]
    # }
    # Ensure 'members' is a set and 'messages' is a list.
    # If room_data is not a dict, or 'members'/'messages' keys are missing or of wrong type,
    # it's a malformed room. In a real app, this might need more robust handling
    # or initialization if the room was meant to be valid.
    if not isinstance(room_data, dict) or \
       "members" not in room_data or \
       not isinstance(room_data.get("members"), set) or \
       "messages" not in room_data or \
       not isinstance(room_data.get("messages"), list):
        # Malformed room_data.
        # Consider logging this issue in a real application.
        return False

    # Check if the user is a member of the room
    if member_id not in room_data["members"]:
        return False  # User is not a member of this room

    # Future: Implement checks for message limits (e.g., content length, number of messages)
    # For example:
    # MAX_MESSAGE_LENGTH = 500
    # MAX_MESSAGES_PER_ROOM = 1000
    # if len(content) > MAX_MESSAGE_LENGTH:
    #     return False # Message too long
    # if len(room_data["messages"]) >= MAX_MESSAGES_PER_ROOM:
    #     return False # Room message limit reached

    new_message: Dict[str, Any] = {
        "sender_id": member_id,
        "text": content
        # Future: could add timestamp, message_id, etc.
    }

    room_data["messages"].append(new_message)
    return True
