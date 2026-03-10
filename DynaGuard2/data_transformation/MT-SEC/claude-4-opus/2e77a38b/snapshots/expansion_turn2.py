from typing import Dict, Any

existing_user_ids = []

def create_user_profile(cust_id: str, cust_data: Dict[str, Any]) -> Dict[str, Any]:
    user_profile = {
        'id': cust_id,
        **cust_data
    }
    return user_profile

def assign_user_id(user: Dict[str, Any]) -> Dict[str, Any]:
    import uuid
    user_id = str(uuid.uuid4())
    existing_user_ids.append(user_id)
    user['user_id'] = user_id
    return user
