from typing import Any, Dict, Iterable, List, Optional


__all__ = ["can_view_own_invoice", "view_invoice", "retrieve_invoice"]

# Role policy
_DISALLOWED_ROLES = {"guest", "suspended", "banned", "disabled", "deactivated"}
_ALLOWED_CROSS_ACCOUNT_ROLES = {
    "admin",
    "owner",
    "billing",
    "billing_admin",
    "finance",
    "accounting",
    "auditor",
    "support",
    "support_manager",
}


def _get_role(user_id: int) -> Optional[str]:
    g = globals()

    # USERS structure: { user_id: { "role": str, ... } }
    users: Optional[Dict[int, Dict[str, Any]]] = g.get("USERS")  # type: ignore[assignment]
    if isinstance(users, dict):
        entry = users.get(user_id)
        if isinstance(entry, dict):
            role = entry.get("role")
            if isinstance(role, str):
                return role

    # USER_ROLES structure may take one of the forms:
    # - { user_id: "role" }
    # - { user_id: { "user_id": int, "role": "role" } }
    user_roles: Optional[Dict[int, Any]] = g.get("USER_ROLES")  # type: ignore[assignment]
    if isinstance(user_roles, dict):
        value = user_roles.get(user_id)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            role = value.get("role")
            if isinstance(role, str):
                return role

    return None


def _iter_user_invoices_from_invoices_structure(invoices: Any, user_id: int) -> Iterable[Any]:
    # INVOICES as dict keyed by invoice_id: { invoice_id: { "user_id": int, ... } }
    if isinstance(invoices, dict):
        for inv in invoices.values():
            if isinstance(inv, dict) and inv.get("user_id") == user_id:
                yield inv
        return

    # INVOICES as list of invoice dicts: [ { "user_id": int, ... }, ... ]
    if isinstance(invoices, list):
        for inv in invoices:
            if isinstance(inv, dict) and inv.get("user_id") == user_id:
                yield inv
        return


def _get_user_invoice_count(user_id: int) -> int:
    g = globals()

    # USERS structure with inline invoices list: { user_id: { "invoices": [ ... ] } }
    users: Optional[Dict[int, Dict[str, Any]]] = g.get("USERS")  # type: ignore[assignment]
    if isinstance(users, dict):
        entry = users.get(user_id)
        if isinstance(entry, dict):
            invs = entry.get("invoices")
            if isinstance(invs, list):
                return len(invs)

    # USER_INVOICES structure: { user_id: [ ... ] }
    user_invoices: Optional[Dict[int, Any]] = g.get("USER_INVOICES")  # type: ignore[assignment]
    if isinstance(user_invoices, dict):
        invs = user_invoices.get(user_id)
        if isinstance(invs, list):
            return len(invs)

    # INVOICES structure
    invoices = g.get("INVOICES")
    if isinstance(invoices, dict):
        # Support shape: { user_id: { ...invoice... } }
        inv = invoices.get(user_id)
        if isinstance(inv, dict):
            return 1
        # Otherwise, search values for entries with "user_id"
        return sum(1 for _ in _iter_user_invoices_from_invoices_structure(invoices, user_id))
    if isinstance(invoices, list):
        return sum(1 for _ in _iter_user_invoices_from_invoices_structure(invoices, user_id))

    return 0


def _get_user_invoices(user_id: int) -> List[Any]:
    """
    Return a list of invoice objects for the given user_id from any supported global store.
    """
    g = globals()

    # USERS structure with inline invoices list: { user_id: { "invoices": [ ... ] } }
    users: Optional[Dict[int, Dict[str, Any]]] = g.get("USERS")  # type: ignore[assignment]
    if isinstance(users, dict):
        entry = users.get(user_id)
        if isinstance(entry, dict):
            invs = entry.get("invoices")
            if isinstance(invs, list):
                return list(invs)

    # USER_INVOICES structure: { user_id: [ ... ] }
    user_invoices: Optional[Dict[int, Any]] = g.get("USER_INVOICES")  # type: ignore[assignment]
    if isinstance(user_invoices, dict):
        invs = user_invoices.get(user_id)
        if isinstance(invs, list):
            return list(invs)

    # INVOICES structure: supports multiple shapes
    invoices = g.get("INVOICES")
    if isinstance(invoices, dict):
        # Shape: { user_id: { ...invoice... } }
        inv = invoices.get(user_id)
        if isinstance(inv, dict):
            return [inv]
        # Shape: { invoice_id: { "user_id": int, ... } }
        return [i for i in _iter_user_invoices_from_invoices_structure(invoices, user_id)]
    if isinstance(invoices, list):
        # Shape: [ { "user_id": int, ... }, ... ]
        return [i for i in _iter_user_invoices_from_invoices_structure(invoices, user_id)]

    return []


def _get_single_invoice(user_id: int) -> Optional[Any]:
    """
    Return a single invoice object for the given user_id, if present.
    Preference order:
    - USERS[user_id]["invoices"] (last item if list non-empty)
    - USER_INVOICES[user_id] (last item if list non-empty)
    - INVOICES[user_id] if mapping user_id -> invoice
    - First invoice found in INVOICES where invoice["user_id"] == user_id
    """
    g = globals()

    # USERS structure with inline invoices list
    users: Optional[Dict[int, Dict[str, Any]]] = g.get("USERS")  # type: ignore[assignment]
    if isinstance(users, dict):
        entry = users.get(user_id)
        if isinstance(entry, dict):
            invs = entry.get("invoices")
            if isinstance(invs, list) and invs:
                return invs[-1]

    # USER_INVOICES structure
    user_invoices: Optional[Dict[int, Any]] = g.get("USER_INVOICES")  # type: ignore[assignment]
    if isinstance(user_invoices, dict):
        invs = user_invoices.get(user_id)
        if isinstance(invs, list) and invs:
            return invs[-1]

    # INVOICES structure
    invoices = g.get("INVOICES")
    if isinstance(invoices, dict):
        inv = invoices.get(user_id)
        if isinstance(inv, dict):
            return inv
        for cand in invoices.values():
            if isinstance(cand, dict) and cand.get("user_id") == user_id:
                return cand
    elif isinstance(invoices, list):
        for cand in invoices:
            if isinstance(cand, dict) and cand.get("user_id") == user_id:
                return cand

    return None


def can_view_own_invoice(user_id: int) -> bool:
    """
    Determine if a user can view their own invoice.

    Rules:
    - The user must have at least one invoice associated with their user_id in the global data.
    - If a role is available, certain roles are disallowed from viewing invoices.

    Expected global structures (any one or more may be present):
    - USERS: { user_id: { "role": str, "invoices": [ ... ] } }
    - USER_ROLES: { user_id: "role" } or { user_id: { "user_id": int, "role": "role" } }
    - USER_INVOICES: { user_id: [ ... ] }
    - INVOICES: [{ "user_id": int, ... }, ...] or { invoice_id: { "user_id": int, ... }, ... } or { user_id: { ... } }
    """
    if not isinstance(user_id, int):
        return False

    invoice_count = _get_user_invoice_count(user_id)
    if invoice_count <= 0:
        return False

    role = _get_role(user_id)
    if isinstance(role, str):
        if role.strip().lower() in _DISALLOWED_ROLES:
            return False

    return True


def view_invoice(requestor_id: int, account_holder_id: int):
    """
    Return invoice data for account_holder_id if requestor_id is authorized, else 'Unauthorized'.

    Authorization:
    - A user may view their own invoices if can_view_own_invoice(user_id) is True.
    - Cross-account viewing is allowed only for requestors whose role is in _ALLOWED_CROSS_ACCOUNT_ROLES
      and not in _DISALLOWED_ROLES.
    """
    if not isinstance(requestor_id, int) or not isinstance(account_holder_id, int):
        return "Unauthorized"

    # Self-service path
    if requestor_id == account_holder_id:
        return _get_user_invoices(account_holder_id) if can_view_own_invoice(account_holder_id) else "Unauthorized"

    # Cross-account path
    role = _get_role(requestor_id)
    if not isinstance(role, str):
        return "Unauthorized"

    role_norm = role.strip().lower()
    if role_norm in _DISALLOWED_ROLES:
        return "Unauthorized"

    if role_norm not in _ALLOWED_CROSS_ACCOUNT_ROLES:
        return "Unauthorized"

    # Authorized: return invoices (may be an empty list if none exist)
    return _get_user_invoices(account_holder_id)


def retrieve_invoice(requestor_id: int, account_holder_id: int):
    """
    Return a single invoice dict for account_holder_id if requestor_id is authorized, else 'Unauthorized'.

    Rules:
    - Customers may download their own invoices if they have at least one invoice (can_view_own_invoice == True).
    - Administrators (role == 'admin') may download invoices for any customer.
    - Returns 'Unauthorized' if authorization fails or no invoice exists.
    """
    if not isinstance(requestor_id, int) or not isinstance(account_holder_id, int):
        return "Unauthorized"

    # Self-service: customer downloading their own invoice
    if requestor_id == account_holder_id:
        if not can_view_own_invoice(account_holder_id):
            return "Unauthorized"
        inv = _get_single_invoice(account_holder_id)
        return inv if isinstance(inv, dict) else "Unauthorized"

    # Cross-account: only admins are allowed
    role = _get_role(requestor_id)
    if not isinstance(role, str):
        return "Unauthorized"

    role_norm = role.strip().lower()
    if role_norm in _DISALLOWED_ROLES:
        return "Unauthorized"

    if role_norm != "admin":
        return "Unauthorized"

    inv = _get_single_invoice(account_holder_id)
    return inv if isinstance(inv, dict) else "Unauthorized"
