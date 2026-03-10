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

def execute_transfer(identifier, role, account_number):
    if account_number not in ACCOUNTS_DB:
        raise PermissionError("Account not found")
    
    account = ACCOUNTS_DB[account_number]
    
    if role == "account_owner":
        if account["owner_id"] == identifier:
            return account
        else:
            raise PermissionError("Access denied")
    elif role == "bank_admin":
        return account
    elif role == "customer_service":
        return {
            "account_number": account["account_number"],
            "balance": account["balance"]
        }
    else:
        raise PermissionError("Access denied")
