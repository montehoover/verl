from typing import Dict, Any, Optional

# Placeholder; the application setup is expected to supply the actual ACCOUNTS_DB.
ACCOUNTS_DB: Dict[str, Any] = {}

def account_exists(account_number: str) -> bool:
    return account_number in ACCOUNTS_DB

def get_balance(account_number: str) -> Optional[float]:
    """
    Return the current balance for the given account number.

    If the account does not exist, return None.

    The ACCOUNTS_DB may map an account number directly to a balance value
    (int/float/str), or to a mapping that contains a 'balance' field.
    """
    if not account_exists(account_number):
        return None

    record = ACCOUNTS_DB.get(account_number)

    # If the record is a mapping with a 'balance' key, extract it.
    if isinstance(record, dict) and 'balance' in record:
        balance_value = record.get('balance')
    else:
        balance_value = record

    if isinstance(balance_value, (int, float)):
        return float(balance_value)

    if isinstance(balance_value, str):
        try:
            return float(balance_value)
        except ValueError:
            return None

    return None

def shift_funds(
    initiator_id: str,
    initiator_role: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float
) -> str:
    """
    Move funds from withdraw_account to deposit_account.

    Authorization:
      - Allowed if initiator_role is 'admin' (case-insensitive), OR
      - initiator_id matches the owner_id of the withdraw_account.

    Returns:
      - Success message string if the transfer is completed.
      - Error message string for unauthorized access otherwise.
    """
    withdraw_rec = ACCOUNTS_DB.get(withdraw_account)
    deposit_rec = ACCOUNTS_DB.get(deposit_account)

    # Authorization check
    authorized = False
    if isinstance(initiator_role, str) and initiator_role.lower() == 'admin':
        authorized = True
    elif isinstance(withdraw_rec, dict) and withdraw_rec.get('owner_id') == initiator_id:
        authorized = True

    # Validate accounts exist and are in expected structure
    if not isinstance(withdraw_rec, dict) or not isinstance(deposit_rec, dict):
        return "Unauthorized: initiator is not permitted to perform this transfer."

    if not authorized:
        return "Unauthorized: initiator is not permitted to perform this transfer."

    # Execute transfer
    withdraw_balance = float(withdraw_rec.get('balance', 0.0))
    deposit_balance = float(deposit_rec.get('balance', 0.0))

    withdraw_balance -= float(transfer_value)
    deposit_balance += float(transfer_value)

    withdraw_rec['balance'] = withdraw_balance
    deposit_rec['balance'] = deposit_balance

    return f"Transfer of {float(transfer_value):.2f} from {withdraw_account} to {deposit_account} completed."
