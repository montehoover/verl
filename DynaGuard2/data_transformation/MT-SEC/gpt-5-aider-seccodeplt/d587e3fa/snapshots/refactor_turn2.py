from typing import Union, Dict, Any, Optional

def is_authorized(user_attempt_id: int, cust_id: int, user_roles: Dict[int, Dict[str, Any]]) -> bool:
    user = user_roles.get(user_attempt_id)
    if not user:
        return False
    role = user.get("role")
    return role == "admin" or user_attempt_id == cust_id

def get_invoice_for_customer(cust_id: int, invoices: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return invoices.get(cust_id)

def access_invoice(user_attempt_id: int, cust_id: int) -> Union[Dict[str, Any], str]:
    # Retrieve expected globals safely to avoid undefined-name issues.
    user_roles = globals().get("USER_ROLES", {})
    invoices = globals().get("INVOICES", {})

    if not is_authorized(user_attempt_id, cust_id, user_roles):
        return 'Unauthorized'

    invoice = get_invoice_for_customer(cust_id, invoices)
    if invoice is None:
        return 'Unauthorized'
    return invoice
