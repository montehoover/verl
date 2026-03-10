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

def execute_transfer(identifier, role, source_acc, destination_acc, value):
    if source_acc not in ACCOUNTS_DB:
        raise PermissionError("Source account not found")
    
    if destination_acc not in ACCOUNTS_DB:
        raise PermissionError("Destination account not found")
    
    source_account = ACCOUNTS_DB[source_acc]
    destination_account = ACCOUNTS_DB[destination_acc]
    
    if role == "account_owner":
        if source_account["owner_id"] != identifier:
            raise PermissionError("Access denied - not the owner of source account")
    elif role == "bank_admin":
        pass
    else:
        raise PermissionError("Access denied - insufficient role permissions")
    
    if source_account["balance"] < value:
        raise PermissionError("Insufficient funds")
    
    source_account["balance"] -= value
    destination_account["balance"] += value
    
    return f"Transfer successful: {value} transferred from {source_acc} to {destination_acc}"
