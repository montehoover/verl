from typing import Dict
from datetime import datetime

def create_message(member_id: str, content: str) -> Dict:
    """
    Creates a message dictionary with member_id, content, and a timestamp.

    Args:
        member_id: The ID of the member sending the message.
        content: The content of the message.

    Returns:
        A dictionary representing the message.
    """
    return {
        "member_id": member_id,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
