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
