import re

def can_view_invoice(user_id: int) -> bool:
    if not isinstance(user_id, int):
        return False

    g = globals()

    role = None
    for name in ("USER_ROLES", "user_roles", "ROLES", "roles"):
        mapping = g.get(name)
        if isinstance(mapping, dict):
            role = mapping.get(user_id)
            break

    if role in ("suspended", "banned", "blocked", "disabled"):
        return False

    permitted_roles = None
    for name in ("PERMITTED_VIEW_OWN_INVOICE_ROLES", "PERMITTED_ROLES", "ALLOWED_ROLES"):
        val = g.get(name)
        if isinstance(val, (set, list, tuple)):
            permitted_roles = set(val)
            break
    if permitted_roles is not None and role not in permitted_roles:
        return False

    for name in ("INVOICES_BY_USER", "USER_INVOICES", "user_invoices"):
        mapping = g.get(name)
        if isinstance(mapping, dict):
            return bool(mapping.get(user_id))

    invoices = None
    for name in ("INVOICES", "invoices"):
        if name in g:
            invoices = g.get(name)
            break
    if invoices is None:
        return False

    if isinstance(invoices, dict):
        for inv in invoices.values():
            if isinstance(inv, dict):
                uid = inv.get("user_id") or inv.get("userId") or inv.get("owner_id")
                if uid == user_id:
                    return True
            elif isinstance(inv, int):
                if inv == user_id:
                    return True
        if user_id in invoices:
            val = invoices[user_id]
            if isinstance(val, (list, tuple, set)):
                return bool(val)
            return True
        return False

    if isinstance(invoices, (list, tuple, set)):
        for inv in invoices:
            if isinstance(inv, dict):
                uid = inv.get("user_id") or inv.get("userId") or inv.get("owner_id")
                if uid == user_id:
                    return True
            elif isinstance(inv, (tuple, list)) and len(inv) >= 2:
                if isinstance(inv[1], int) and inv[1] == user_id:
                    return True
        return False

    return False

def format_invoice_amount(amount: str) -> float:
    if not isinstance(amount, str):
        raise TypeError("amount must be a string")

    s = amount.strip()
    neg = False

    # Parentheses indicate negative amount, e.g., "(123.45)"
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # Keep only digits, decimal point, comma and minus
    s = re.sub(r"[^0-9,\.-]", "", s)

    # Determine negativity based on minus signs present
    if "-" in s:
        if s.count("-") % 2 == 1:
            neg = not neg
        s = s.replace("-", "")

    # Remove thousands separators
    s = s.replace(",", "")

    if not s or s == ".":
        return -0.0 if neg else 0.0

    try:
        value = float(s)
    except ValueError:
        # Fallback: extract first numeric segment
        m = re.search(r"\d+(?:\.\d+)?", s)
        if not m:
            return -0.0 if neg else 0.0
        value = float(m.group(0))

    return -value if neg else value

def download_invoice(current_user_id: int, customer_id: int):
    if not isinstance(current_user_id, int) or not isinstance(customer_id, int):
        return "Unauthorized"

    g = globals()
    roles = g.get("USER_ROLES")
    invoices = g.get("INVOICES")

    if not isinstance(roles, dict) or not isinstance(invoices, dict):
        return "Unauthorized"

    current_info = roles.get(current_user_id)
    if current_info is None:
        return "Unauthorized"

    if isinstance(current_info, dict):
        role = current_info.get("role") or current_info.get("name") or current_info.get("type")
    elif isinstance(current_info, str):
        role = current_info
    else:
        role = None

    if role == "admin" or (role == "customer" and current_user_id == customer_id):
        return invoices.get(customer_id)

    return "Unauthorized"
