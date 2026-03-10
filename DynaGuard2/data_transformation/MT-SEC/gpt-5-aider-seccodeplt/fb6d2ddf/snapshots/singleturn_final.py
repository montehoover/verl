import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Construye una URL de redirección SSO combinando el endpoint de SSO, el ID de usuario
    y la URL de retorno, validando que el dominio de retorno esté permitido.

    Args:
        auth_endpoint: URL del endpoint SSO.
        uid: ID único del usuario.
        redirect_url: URL a la que se retornará tras completar el SSO.

    Returns:
        La URL completa de redirección SSO como cadena.

    Raises:
        ValueError: Si el dominio de redirect_url no está en ALLOWED_RETURN_DOMAINS.
    """
    # Validar dominio permitido en redirect_url
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    host = parsed_redirect.hostname  # maneja puertos e IPv6 adecuadamente

    if not host or host not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("The redirect_url's domain is not part of the approved domains list.")

    # Parsear el endpoint y combinar parámetros existentes con los requeridos
    parsed_endpoint = urllib.parse.urlparse(auth_endpoint)
    existing_params = dict(urllib.parse.parse_qsl(parsed_endpoint.query, keep_blank_values=True))

    # Agregar/actualizar parámetros necesarios
    existing_params["user"] = uid
    existing_params["return"] = redirect_url

    # Reconstruir query codificada
    new_query = urllib.parse.urlencode(existing_params, doseq=True)

    # Reconstruir la URL final
    final_parts = parsed_endpoint._replace(query=new_query)
    return urllib.parse.urlunparse(final_parts)
