from typing import Dict, Any

def create_user_profile(cust_id: str, cust_data: Dict[str, Any]) -> Dict[str, Any]:
    user_profile = {
        'id': cust_id,
        **cust_data
    }
    return user_profile
