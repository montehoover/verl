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


def compose(*funcs):
    """
    Compose multiple single-argument functions into a pipeline.
    The output of each function is passed as the input to the next.
    """
    def _composed(value):
        for f in funcs:
            value = f(value)
        return value
    return _composed


# Pure functions

def authorize_transfer(account_holder: str, user_role: str, withdrawal_owner_id: str) -> None:
    """
    Pure authorization check. Raises PermissionError on unauthorized.
    """
    if user_role == "bank_admin":
        return
    if user_role == "account_owner" and withdrawal_owner_id == account_holder:
        return
    raise PermissionError("Unauthorized access: user is not permitted to transfer from this account.")


def withdraw(current_balance: float, amount: float) -> float:
    """
    Pure withdrawal. Returns the new balance; raises ValueError if insufficient funds.
    """
    if current_balance < amount:
        raise ValueError("Insufficient funds in the withdrawal account.")
    return current_balance - amount


def deposit(current_balance: float, amount: float) -> float:
    """
    Pure deposit. Returns the new balance.
    """
    return current_balance + amount


def send_funds(
    account_holder: str,
    user_role: str,
    withdrawal_account: str,
    receiving_account: str,
    fund_amount: float
) -> str:
    """
    Move money from one account to another with authorization checks.

    Args:
        account_holder: User ID initiating the transaction.
        user_role: Role of the user ('account_owner', 'bank_admin', 'customer_service').
        withdrawal_account: Account number to withdraw funds from.
        receiving_account: Account number to deposit funds into.
        fund_amount: Amount of funds to transfer.

    Returns:
        Success message confirming the transfer if authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If input is invalid or funds are insufficient.
        TypeError: If fund_amount is not a number.
    """
    # Basic validations
    if not isinstance(fund_amount, (int, float)):
        raise TypeError("fund_amount must be a number.")
    if fund_amount <= 0:
        raise ValueError("fund_amount must be greater than 0.")
    if withdrawal_account == receiving_account:
        raise ValueError("withdrawal_account and receiving_account must be different.")
    if withdrawal_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdrawal account '{withdrawal_account}' does not exist.")
    if receiving_account not in ACCOUNTS_DB:
        raise ValueError(f"Receiving account '{receiving_account}' does not exist.")

    withdrawal_acc = ACCOUNTS_DB[withdrawal_account]
    receiving_acc = ACCOUNTS_DB[receiving_account]

    # Build a pure pipeline over a context object
    context = {
        "account_holder": account_holder,
        "user_role": user_role,
        "withdrawal_acc": withdrawal_acc,
        "receiving_acc": receiving_acc,
        "amount": float(fund_amount),
        "withdrawal_new_balance": None,
        "receiving_new_balance": None,
    }

    def step_authorize(ctx):
        authorize_transfer(
            ctx["account_holder"],
            ctx["user_role"],
            ctx["withdrawal_acc"]["owner_id"],
        )
        return ctx

    def step_withdraw(ctx):
        ctx["withdrawal_new_balance"] = withdraw(
            ctx["withdrawal_acc"]["balance"],
            ctx["amount"],
        )
        return ctx

    def step_deposit(ctx):
        ctx["receiving_new_balance"] = deposit(
            ctx["receiving_acc"]["balance"],
            ctx["amount"],
        )
        return ctx

    pipeline = compose(
        step_authorize,
        step_withdraw,
        step_deposit,
    )

    result_ctx = pipeline(context)

    # Apply state changes after pure computation succeeds
    ACCOUNTS_DB[withdrawal_account]["balance"] = result_ctx["withdrawal_new_balance"]
    ACCOUNTS_DB[receiving_account]["balance"] = result_ctx["receiving_new_balance"]

    return f"Transferred {fund_amount:.2f} from {withdrawal_account} to {receiving_account}."
