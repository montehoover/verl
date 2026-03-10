"""
Banking transfer system module.

This module provides functionality for transferring funds between bank accounts
with proper authorization checks and balance validation.
"""

# Sample database of accounts for demonstration purposes
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


def check_authorization(initiator_id: str, position: str, account_owner_id: str) -> bool:
    """
    Pure function to check if a user is authorized to perform a transfer.
    
    Authorization rules:
    - Bank administrators and customer service representatives can transfer from any account
    - Account owners can only transfer from their own accounts
    - All other positions are denied access
    
    Args:
        initiator_id: ID of the user initiating the transfer
        position: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        account_owner_id: ID of the account owner from which funds will be withdrawn
    
    Returns:
        bool: True if authorized, False otherwise
    """
    # Admin and customer service roles have universal transfer privileges
    if position in ['bank_admin', 'customer_service']:
        return True
    
    # Account owners can only transfer from their own accounts
    elif position == 'account_owner' and initiator_id == account_owner_id:
        return True
    
    # All other cases are unauthorized
    return False


def calculate_updated_balances(
    withdraw_balance: float, 
    deposit_balance: float, 
    transfer_amount: float
) -> tuple[float, float]:
    """
    Pure function to calculate updated account balances after a transfer.
    
    This function performs the mathematical calculation for transferring funds
    without modifying any external state.
    
    Args:
        withdraw_balance: Current balance of the withdrawal account
        deposit_balance: Current balance of the deposit account
        transfer_amount: Amount to transfer (must be positive)
    
    Returns:
        tuple: (new_withdraw_balance, new_deposit_balance)
    """
    # Calculate the new balance after withdrawal
    new_withdraw_balance = withdraw_balance - transfer_amount
    
    # Calculate the new balance after deposit
    new_deposit_balance = deposit_balance + transfer_amount
    
    return new_withdraw_balance, new_deposit_balance


def initiate_transfer(
    initiator_id: str, 
    position: str, 
    withdraw_account: str, 
    deposit_account: str, 
    transfer_value: float
) -> str:
    """
    Facilitate money transfers between two accounts within the banking system.
    
    This function handles the complete transfer process including:
    1. Account existence validation
    2. Authorization verification
    3. Balance sufficiency check
    4. Actual transfer execution
    
    Args:
        initiator_id: ID of the user initiating the transfer
        position: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        withdraw_account: Account number from which funds will be withdrawn
        deposit_account: Target account number where funds will be deposited
        transfer_value: Amount to be transferred (must be positive)
    
    Returns:
        str: Success message confirming the transfer when authorized,
             otherwise an error message indicating the reason for failure
    
    Raises:
        PermissionError: When the user lacks authorization to perform the transfer
    """
    # Validate that both accounts exist in the system
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdraw_account} not found"
    
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} not found"
    
    # Retrieve account details from the database
    withdraw_acc = ACCOUNTS_DB[withdraw_account]
    deposit_acc = ACCOUNTS_DB[deposit_account]
    
    # Verify user authorization for this transfer
    authorized = check_authorization(
        initiator_id, 
        position, 
        withdraw_acc['owner_id']
    )
    
    if not authorized:
        # Raise exception for unauthorized access attempts
        raise PermissionError(
            f"User {initiator_id} with position {position} is not authorized "
            f"to transfer from account {withdraw_account}"
        )
    
    # Ensure the withdrawal account has sufficient funds
    if withdraw_acc['balance'] < transfer_value:
        return f"Error: Insufficient balance in account {withdraw_account}"
    
    # Calculate new balances using the pure function
    new_withdraw_balance, new_deposit_balance = calculate_updated_balances(
        withdraw_acc['balance'], 
        deposit_acc['balance'], 
        transfer_value
    )
    
    # Execute the transfer by updating account balances
    withdraw_acc['balance'] = new_withdraw_balance
    deposit_acc['balance'] = new_deposit_balance
    
    # Return success confirmation with transfer details
    return (
        f"Transfer successful: ${transfer_value} transferred "
        f"from {withdraw_account} to {deposit_account}"
    )
