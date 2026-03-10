from typing import Dict, Any

accounts = {}

def create_account_info(account_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'account_id': account_id,
        **account_data
    }

def update_account_info(account_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
    if account_id in accounts:
        accounts[account_id].update(new_data)
    else:
        accounts[account_id] = {
            'account_id': account_id,
            **new_data
        }
    return accounts[account_id]
