chat_rooms = {}

# Limits (can be adjusted as needed)
MAX_MESSAGE_LENGTH = 5000
MAX_MESSAGES_PER_ROOM = 10000


def get_user_messages(room_name: str, member_id: str) -> list:
    """
    Return a list of messages sent by member_id within the chat room identified by room_name.
    - If the user is not a member of the room or the room doesn't exist, return an empty list.

    Supported room structures:
    1) A mapping of user_id -> messages (list/tuple or other iterable/single message)
       Example:
         chat_rooms = {
             "room1": {
                 "userA": ["hi", "hello"],
                 "userB": ["hey"]
             }
         }
    2) A mapping containing 'members' and 'messages':
       - 'members' can be a set/list/tuple/dict-like of user IDs
       - 'messages' is a mapping of user_id -> messages
       Example:
         chat_rooms = {
             "room1": {
                 "members": {"userA", "userB"},
                 "messages": {
                     "userA": ["hi", "hello"],
                     "userB": ["hey"]
                 }
             }
         }
    """
    room = chat_rooms.get(room_name)
    if room is None:
        return []

    # Helper to coerce a messages value into a list
    def to_list(msgs):
        if msgs is None:
            return []
        if isinstance(msgs, list):
            return list(msgs)
        if isinstance(msgs, tuple):
            return list(msgs)
        if isinstance(msgs, (str, bytes)):
            return [msgs]
        try:
            return list(msgs)
        except TypeError:
            return [msgs]

    # Case 1: room is a mapping user_id -> messages
    if isinstance(room, dict):
        if member_id in room and member_id not in ("members", "messages"):
            return to_list(room.get(member_id))

        # Case 2: room has explicit 'members' and 'messages' sections
        members = room.get("members")
        messages_map = room.get("messages")

        # Determine membership
        is_member = False
        if isinstance(members, (set, list, tuple)):
            is_member = member_id in members
        elif isinstance(members, dict):
            is_member = member_id in members
        elif hasattr(members, "__contains__") and not isinstance(members, (str, bytes)):
            try:
                is_member = member_id in members
            except Exception:
                is_member = False

        if is_member:
            if isinstance(messages_map, dict):
                return to_list(messages_map.get(member_id, []))
            return []

    # Unsupported room structure; cannot determine membership/messages
    return []


def push_text_to_room(room_name: str, member_id: str, content: str) -> bool:
    """
    Append a message to the chat room's history when within limits.
    Returns True if appended; otherwise, False.

    Behavior:
    - Creates the room if it doesn't exist.
    - Supports multiple room structures:
      a) {'messages': list, 'members': <optional container>}
         - Appends {'member_id': <id>, 'content': <text>} to 'messages' list.
      b) {'messages': {user_id: list_of_texts}, 'members': <optional container>}
         - Appends <text> to messages[user_id] list.
      c) {user_id: list_of_texts, ...} (no 'messages' key)
         - Appends <text> to mapping under user_id.
      d) [ ... ] (room as a list of messages)
         - Appends {'member_id': <id>, 'content': <text>} to the list.

    Limits enforced:
    - MAX_MESSAGE_LENGTH for content length.
    - MAX_MESSAGES_PER_ROOM for total number of messages in the room (best-effort for mappings).
    """
    # Validate inputs
    if not isinstance(room_name, str) or not isinstance(member_id, str) or not isinstance(content, str):
        return False
    if not room_name or not member_id:
        return False
    # Reject empty/whitespace-only messages
    if content.strip() == "":
        return False
    if len(content) > MAX_MESSAGE_LENGTH:
        return False

    # Ensure the room exists
    room = chat_rooms.get(room_name)
    if room is None:
        room = {"members": set(), "messages": []}
        chat_rooms[room_name] = room

    # Helper: count total messages in a room (best-effort across shapes)
    def count_messages(r) -> int:
        try:
            if isinstance(r, dict):
                if "messages" in r:
                    msgs = r.get("messages")
                    if isinstance(msgs, (list, tuple)):
                        return len(msgs)
                    if isinstance(msgs, dict):
                        total = 0
                        for v in msgs.values():
                            if isinstance(v, (list, tuple)):
                                total += len(v)
                            elif v is None:
                                continue
                            else:
                                # Treat singletons/iterables as 1 if not list/tuple
                                try:
                                    iter(v)
                                    total += len(list(v))  # may be expensive, fallback to 1 on error
                                except Exception:
                                    total += 1
                        return total
                # Mapping user_id -> messages without 'messages' key
                total = 0
                for k, v in r.items():
                    if k in ("members", "messages"):
                        continue
                    if isinstance(v, (list, tuple)):
                        total += len(v)
                    elif v is None:
                        continue
                    else:
                        try:
                            iter(v)
                            total += len(list(v))
                        except Exception:
                            total += 1
                return total
            if isinstance(r, (list, tuple)):
                return len(r)
            # Unknown container; fallback to 0 or len if available
            try:
                return len(r)  # type: ignore
            except Exception:
                return 0
        except Exception:
            return 0

    # Enforce per-room message limit
    current_count = count_messages(room)
    if current_count >= MAX_MESSAGES_PER_ROOM:
        return False

    # Helper: add member_id into a 'members' container if present
    def add_member_if_possible(container):
        try:
            if container is None:
                return
            if isinstance(container, set):
                container.add(member_id)
                return
            if isinstance(container, list):
                if member_id not in container:
                    container.append(member_id)
                return
            if isinstance(container, dict):
                container.setdefault(member_id, True)
                return
            # For other containers that support 'add' or 'append'
            if hasattr(container, "add"):
                try:
                    container.add(member_id)  # type: ignore
                    return
                except Exception:
                    pass
            if hasattr(container, "append"):
                try:
                    if member_id not in container:
                        container.append(member_id)  # type: ignore
                    return
                except Exception:
                    pass
        except Exception:
            return

    # Prepare message record for list-based histories
    msg_record = {"member_id": member_id, "content": content}

    # Append according to structure
    try:
        if isinstance(room, dict):
            # Case: {'messages': list} or {'messages': dict}
            if "messages" in room:
                msgs = room.get("messages")

                # messages: list of message records
                if isinstance(msgs, list):
                    room["messages"].append(msg_record)
                    add_member_if_possible(room.get("members"))
                    return True

                # messages: dict mapping user_id -> list of texts
                if isinstance(msgs, dict):
                    lst = msgs.get(member_id)
                    if lst is None:
                        lst = []
                        msgs[member_id] = lst
                    elif not isinstance(lst, list):
                        # Coerce to list preserving existing content
                        if lst is None:
                            lst = []
                        elif isinstance(lst, tuple):
                            lst = list(lst)
                        elif isinstance(lst, (str, bytes)):
                            lst = [lst]
                        else:
                            try:
                                lst = list(lst)
                            except TypeError:
                                lst = [lst]
                        msgs[member_id] = lst
                    lst.append(content)
                    add_member_if_possible(room.get("members"))
                    return True

            # Case: mapping user_id -> messages, without 'messages' key
            if member_id not in ("members", "messages"):
                slot = room.get(member_id)
                if slot is None:
                    room[member_id] = [content]
                    add_member_if_possible(room.get("members"))
                    return True
                if isinstance(slot, list):
                    slot.append(content)
                    add_member_if_possible(room.get("members"))
                    return True
                # Coerce various types into a list and append
                if isinstance(slot, tuple):
                    room[member_id] = list(slot) + [content]
                    add_member_if_possible(room.get("members"))
                    return True
                if isinstance(slot, (str, bytes)):
                    room[member_id] = [slot, content]
                    add_member_if_possible(room.get("members"))
                    return True
                try:
                    room[member_id] = list(slot) + [content]
                    add_member_if_possible(room.get("members"))
                    return True
                except Exception:
                    room[member_id] = [content]
                    add_member_if_possible(room.get("members"))
                    return True

            # Fallback: if 'members' key exists but no messages container we understand,
            # initialize a list-based history.
            room.setdefault("messages", [])
            if isinstance(room["messages"], list):
                room["messages"].append(msg_record)
                add_member_if_possible(room.get("members"))
                return True
            return False

        # Room is a list-like: append message record
        if isinstance(room, list):
            room.append(msg_record)
            return True

        if hasattr(room, "append"):
            try:
                room.append(msg_record)  # type: ignore
                return True
            except Exception:
                return False

        # Unsupported room container
        return False
    except Exception:
        return False
