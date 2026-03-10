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

def execute_transfer(identifier: str, role: str, account_number: str):
    """
    Allows users to view account details based on their role and logs access attempts.

    Args:
        identifier (str): The user ID.
        role (str): The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_number (str): The account number to access.

    Returns:
        dict or float: Account details or balance based on permissions.

    Raises:
        PermissionError: If the user is not authorized to view the details.
        KeyError: If the account_number does not exist.
    """
    timestamp = datetime.datetime.now()
    log_entry = {
        "timestamp": timestamp,
        "identifier": identifier,
        "role": role,
        "account_number": account_number,
        "status": "",
        "message": ""
    }

    if account_number not in ACCOUNTS_DB:
        log_entry["status"] = "FAILURE"
        log_entry["message"] = f"Account {account_number} not found."
        ACCESS_LOG.append(log_entry)
        raise KeyError(log_entry["message"])

    account_info = ACCOUNTS_DB[account_number]

    if role == "account_owner":
        if account_info["owner_id"] == identifier:
            log_entry["status"] = "SUCCESS"
            log_entry["message"] = "Accessed full account details as owner."
            ACCESS_LOG.append(log_entry)
            return account_info
        else:
            log_entry["status"] = "FAILURE"
            log_entry["message"] = f"User {identifier} is not the owner of account {account_number}."
            ACCESS_LOG.append(log_entry)
            raise PermissionError(log_entry["message"])
    elif role == "bank_admin":
        log_entry["status"] = "SUCCESS"
        log_entry["message"] = "Accessed full account details as bank admin."
        ACCESS_LOG.append(log_entry)
        return account_info
    elif role == "customer_service":
        log_entry["status"] = "SUCCESS"
        log_entry["message"] = "Accessed account balance as customer service."
        ACCESS_LOG.append(log_entry)
        return {"balance": account_info["balance"]}
    else:
        log_entry["status"] = "FAILURE"
        log_entry["message"] = f"Role {role} is not authorized to view account details."
        ACCESS_LOG.append(log_entry)
        raise PermissionError(log_entry["message"])
