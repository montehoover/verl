# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

def get_user_phone(request_data):
    user_id = request_data.get("body", {}).get("user_id", "")
    if user_id in db_store.get("users", {}):
        return db_store["users"][user_id].get("cell", "")
    return ""
