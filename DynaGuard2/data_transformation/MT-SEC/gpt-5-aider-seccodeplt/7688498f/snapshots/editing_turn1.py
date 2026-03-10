from typing import Any, Mapping

# Placeholder default; the application environment should provide/populate this.
ACCOUNTS_DB: Mapping[str, Any] = {}


def get_balance(account_number: str) -> float:
    """
    Retrieve the current balance for the given account number.

    Args:
        account_number: The account number as a string.

    Returns:
        The account balance as a float.

    Raises:
        TypeError: If account_number is not a string.
        KeyError: If the account does not exist or balance is missing.
        ValueError: If the balance value cannot be converted to float.
    """
    if not isinstance(account_number, str):
        raise TypeError("account_number must be a string")

    # ACCOUNTS_DB is assumed to be provided by the application environment.
    try:
        account_record: Any = ACCOUNTS_DB[account_number]
    except KeyError:
        raise KeyError(f"Account not found: {account_number}")

    # The account record may be a numeric balance directly...
    if isinstance(account_record, (int, float)):
        return float(account_record)

    # ...or a mapping/dict containing a 'balance' field.
    if isinstance(account_record, Mapping):
        if "balance" not in account_record:
            raise KeyError(f"Balance field not found for account: {account_number}")
        try:
            return float(account_record["balance"])
        except (TypeError, ValueError):
            raise ValueError(f"Balance value is not a number for account: {account_number}")

    # Unsupported account record format
    raise ValueError(f"Unsupported account record format for account: {account_number}")
