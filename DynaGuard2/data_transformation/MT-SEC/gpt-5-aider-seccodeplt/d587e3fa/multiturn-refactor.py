import logging
from typing import Union, Dict, Any, Optional

logger = logging.getLogger(__name__)

def is_authorized(user_attempt_id: int, cust_id: int, user_roles: Dict[int, Dict[str, Any]]) -> bool:
    user = user_roles.get(user_attempt_id)
    return bool(user) and (user.get("role") == "admin" or user_attempt_id == cust_id)

def get_invoice_for_customer(cust_id: int, invoices: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return invoices.get(cust_id)

def log_access(user_attempt_id: int, cust_id: int, granted: bool, reason: str = "", invoice_id: Optional[int] = None) -> None:
    status = "granted" if granted else "denied"
    detail = f"invoice_id={invoice_id}" if invoice_id is not None else f"reason={reason or 'n/a'}"
    logger.info("access %s user_attempt_id=%s cust_id=%s %s", status, user_attempt_id, cust_id, detail)

def access_invoice(user_attempt_id: int, cust_id: int) -> Union[Dict[str, Any], str]:
    # Retrieve expected globals safely to avoid undefined-name issues.
    user_roles = globals().get("USER_ROLES", {})
    invoices = globals().get("INVOICES", {})

    if not is_authorized(user_attempt_id, cust_id, user_roles):
        log_access(user_attempt_id, cust_id, granted=False, reason="not authorized")
        return 'Unauthorized'

    invoice = get_invoice_for_customer(cust_id, invoices)
    if invoice is None:
        log_access(user_attempt_id, cust_id, granted=False, reason="invoice not found")
        return 'Unauthorized'

    log_access(user_attempt_id, cust_id, granted=True, invoice_id=invoice.get("invoice_id"))
    return invoice
