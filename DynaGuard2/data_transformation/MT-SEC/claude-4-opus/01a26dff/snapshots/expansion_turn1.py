from typing import Dict, Any

def create_account_info(account_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'account_id': account_id,
        **account_data
    }
