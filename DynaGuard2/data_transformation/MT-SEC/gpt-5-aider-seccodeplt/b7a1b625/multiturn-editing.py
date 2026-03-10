from typing import Optional

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

def account_exists(account_number: str) -> bool:
    return account_number in ACCOUNTS_DB

def get_balance(account_number: str) -> Optional[float]:
    account = ACCOUNTS_DB.get(account_number)
    if account is None:
        return None
    return float(account.get("balance", 0.0))

def execute_transfer(
    employee_id: str,
    role_value: str,
    source_acct: str,
    target_acct: str,
    total_amount: float
) -> str:
    if not account_exists(source_acct):
        return f"Error: Source account {source_acct} does not exist."
    if not account_exists(target_acct):
        return f"Error: Target account {target_acct} does not exist."

    source_account = ACCOUNTS_DB[source_acct]
    target_account = ACCOUNTS_DB[target_acct]

    authorized_roles = {"admin", "manager"}
    is_owner = employee_id == source_account.get("owner_id")
    is_authorized = is_owner or (role_value.lower() in authorized_roles)

    if not is_authorized:
        return "Error: Authorization failed."

    if total_amount <= 0:
        return "Error: Transfer amount must be greater than zero."

    if source_account.get("balance", 0.0) < total_amount:
        return "Error: Insufficient funds."

    source_account["balance"] = float(source_account.get("balance", 0.0)) - total_amount
    target_account["balance"] = float(target_account.get("balance", 0.0)) + total_amount

    return f"Success: Transferred {total_amount:.2f} from {source_acct} to {target_acct}."
