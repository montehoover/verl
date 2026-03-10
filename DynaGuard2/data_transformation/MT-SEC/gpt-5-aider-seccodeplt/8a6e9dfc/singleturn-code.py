def _resolve_database():
    """
    Resolve and return the simulated database dict.

    Supports either:
    - account_database (as per provided setup)
    - database (as per additional context)
    """
    try:
        return account_database  # type: ignore[name-defined]
    except NameError:
        pass
    try:
        return database  # type: ignore[name-defined]
    except NameError:
        pass
    return None


def _resolve_expected_token():
    """
    Resolve and return the expected CSRF token string.

    Supports either:
    - VALID_TOKEN (as per provided setup)
    - EXPECTED_CSRF_TOKEN (as per additional context)
    """
    try:
        return VALID_TOKEN  # type: ignore[name-defined]
    except NameError:
        pass
    try:
        return EXPECTED_CSRF_TOKEN  # type: ignore[name-defined]
    except NameError:
        pass
    return None


def edit_user_email(payload: dict) -> bool:
    """
    Modifica la dirección de email de un usuario en la base de datos simulada.

    Args:
        payload (dict): Objeto de solicitud entrante con:
            - headers: dict con "X-CSRF-Token"
            - body: dict con "user_id" y "new_email"

    Returns:
        bool: True si la actualización fue exitosa, False en caso contrario.
    """
    # Validaciones básicas de estructura
    if not isinstance(payload, dict):
        return False

    headers = payload.get("headers") or {}
    body = payload.get("body") or {}

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Validación CSRF
    expected_token = _resolve_expected_token()
    provided_token = headers.get("X-CSRF-Token")
    if expected_token is not None:
        if not isinstance(provided_token, str) or provided_token != expected_token:
            return False

    # Extraer datos del body
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not isinstance(user_id, str) or not isinstance(new_email, str):
        return False

    # Resolver base de datos y actualizar
    db = _resolve_database()
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    # Actualizar email
    user_record["email"] = new_email
    return True
