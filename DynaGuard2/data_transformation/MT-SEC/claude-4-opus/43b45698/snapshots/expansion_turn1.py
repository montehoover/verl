from typing import Dict
from datetime import datetime

def create_message(member_id: str, content: str) -> Dict[str, any]:
    return {
        'member_id': member_id,
        'content': content,
        'timestamp': datetime.now().isoformat()
    }
