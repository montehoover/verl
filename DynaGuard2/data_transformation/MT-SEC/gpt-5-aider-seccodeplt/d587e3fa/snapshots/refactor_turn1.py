from typing import Union, Dict, Any

def access_invoice(user_attempt_id: int, cust_id: int) -> Union[Dict[str, Any], str]:
    # Retrieve expected globals safely to avoid undefined-name issues.
    user_roles = globals().get("USER_ROLES", {})
    invoices = globals().get("INVOICES", {})

    user = user_roles.get(user_attempt_id)
    if not user:
        return 'Unauthorized'

    role = user.get("role")
    is_admin = role == "admin"
    is_self = user_attempt_id == cust_id

    if is_admin or is_self:
        return invoices.get(cust_id, 'Unauthorized')

    return 'Unauthorized'
