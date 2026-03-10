from typing import Dict, Any

USERS: Dict[str, Dict[str, Any]] = {}
user_cache: Dict[str, Dict[str, Any]] = {}


def add_user(user_id: str, user_info: Dict[str, Any]) -> bool:
    if not isinstance(user_id, str) or not isinstance(user_info, dict):
        return False
    if user_id == "":
        return False
    if user_id in USERS:
        return False
    USERS[user_id] = user_info
    return True


def store_customer_info(cust_id: str, cust_data: Dict[str, Any]) -> bool:
    if not isinstance(cust_id, str) or not isinstance(cust_data, dict):
        return False
    if cust_id == "":
        return False
    user_cache[cust_id] = cust_data
    return True
