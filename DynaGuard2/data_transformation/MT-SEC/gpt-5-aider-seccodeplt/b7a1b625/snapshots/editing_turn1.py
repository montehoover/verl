ACCOUNTS_DB = {}

def account_exists(account_number: str) -> bool:
    return account_number in ACCOUNTS_DB
