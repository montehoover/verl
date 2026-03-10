from typing import Dict
from datetime import datetime


def create_message(member_id: str, content: str) -> Dict[str, str]:
    """
    Create a message dictionary with member_id, content, and a UTC timestamp.
    """
    return {
        "member_id": member_id,
        "content": content,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
