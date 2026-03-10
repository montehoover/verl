import datetime

ACCOUNTS_DB = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": 1000.0
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": 500.0
    }
}

ACCESS_LOG = []

def execute_transfer(identifier, role, account_number):
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "user_id": identifier,
        "role": role,
        "account_number": account_number,
        "success": False,
        "reason": None
    }
    
    try:
        if account_number not in ACCOUNTS_DB:
            log_entry["reason"] = "Account not found"
            ACCESS_LOG.append(log_entry)
            raise PermissionError("Account not found")
        
        account = ACCOUNTS_DB[account_number]
        
        if role == "account_owner":
            if account["owner_id"] == identifier:
                log_entry["success"] = True
                ACCESS_LOG.append(log_entry)
                return account
            else:
                log_entry["reason"] = "Not account owner"
                ACCESS_LOG.append(log_entry)
                raise PermissionError("Access denied")
        elif role == "bank_admin":
            log_entry["success"] = True
            ACCESS_LOG.append(log_entry)
            return account
        elif role == "customer_service":
            log_entry["success"] = True
            ACCESS_LOG.append(log_entry)
            return {
                "account_number": account["account_number"],
                "balance": account["balance"]
            }
        else:
            log_entry["reason"] = "Invalid role"
            ACCESS_LOG.append(log_entry)
            raise PermissionError("Access denied")
    except Exception as e:
        if not any(entry["timestamp"] == timestamp for entry in ACCESS_LOG):
            ACCESS_LOG.append(log_entry)
        raise
