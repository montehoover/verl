from typing import Any, Dict, Iterable, List, Optional


__all__ = ["can_view_own_invoice", "view_invoice"]

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

    # USER_ROLES structure: { user_id: str }
    user_roles: Optional[Dict[int, Any]] = g.get("USER_ROLES")  # type: ignore[assignment]
    if isinstance(user_roles, dict):
        role = user_roles.get(user_id)
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

    # INVOICES structure: list/dict containing invoices with "user_id"
    invoices = g.get("INVOICES")
    if invoices is not None:
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

    # INVOICES structure: list/dict containing invoices with "user_id"
    invoices = g.get("INVOICES")
    if invoices is not None:
        return [inv for inv in _iter_user_invoices_from_invoices_structure(invoices, user_id)]

    return []


def can_view_own_invoice(user_id: int) -> bool:
    """
    Determine if a user can view their own invoice.

    Rules:
    - The user must have at least one invoice associated with their user_id in the global data.
    - If a role is available, certain roles are disallowed from viewing invoices.

    Expected global structures (any one or more may be present):
    - USERS: { user_id: { "role": str, "invoices": [ ... ] } }
    - USER_ROLES: { user_id: "role" }
    - USER_INVOICES: { user_id: [ ... ] }
    - INVOICES: [{ "user_id": int, ... }, ...] or { invoice_id: { "user_id": int, ... }, ... }
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
