import urllib.parse
import logging

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _mask_secret(value: str, keep: int = 2) -> str:
    """
    Mask a secret value for logging, keeping a small prefix/suffix for correlation.
    Example: token -> to***en
    """
    if not isinstance(value, str):
        return "<non-string>"
    n = len(value)
    if n <= keep * 2:
        return "*" * n if n else ""
    return f"{value[:keep]}{'*' * (n - (keep * 2))}{value[-keep:]}"


def _resolve_callback(application_url: str, callback_route: str) -> tuple[urllib.parse.ParseResult, urllib.parse.ParseResult]:
    """
    Pure function: Resolves the callback_route into an absolute URL using application_url when needed.

    Returns:
        (app_parsed, full_callback_parsed)
    Raises:
        ValueError: If callback_route is relative and application_url is not an absolute URL.
    """
    logger.debug(
        "Resolving callback: application_url=%s, callback_route=%s",
        application_url,
        callback_route,
    )

    app_parsed = urllib.parse.urlparse(application_url)
    route_parsed = urllib.parse.urlparse(callback_route)

    if route_parsed.scheme or route_parsed.netloc:
        # Absolute URL provided as callback_route
        full_callback_parsed = route_parsed
        logger.debug("Callback route is absolute. Using as-is: %s", full_callback_parsed.geturl())
    else:
        # Relative path; resolve against the application_url
        if not app_parsed.scheme or not app_parsed.netloc:
            logger.error(
                "application_url must be an absolute URL when callback_route is relative. "
                "application_url=%s, callback_route=%s",
                application_url,
                callback_route,
            )
            raise ValueError("application_url must be an absolute URL when callback_route is relative")
        full_callback_parsed = urllib.parse.urlparse(
            urllib.parse.urljoin(application_url, callback_route)
        )
        logger.debug("Resolved relative callback against application URL: %s", full_callback_parsed.geturl())

    return app_parsed, full_callback_parsed


def _validate_callback_domain(
    app_parsed: urllib.parse.ParseResult,
    full_callback_parsed: urllib.parse.ParseResult,
    allowed_domains: set[str] = ALLOWED_CALLBACK_DOMAINS,
) -> None:
    """
    Pure function: Validates that the callback URL's domain is either the same as the application's
    domain or is in the allowed domains set.

    Raises:
        ValueError: If the callback domain cannot be determined or is not allowed.
    """
    callback_host = (full_callback_parsed.hostname or "").lower()
    app_host = (app_parsed.hostname or "").lower()

    logger.debug(
        "Validating callback domain. app_host=%s, callback_host=%s, allowed_domains=%s",
        app_host,
        callback_host,
        allowed_domains,
    )

    if not callback_host:
        logger.error("Unable to determine callback URL domain. Parsed URL=%s", full_callback_parsed.geturl())
        raise ValueError("Unable to determine callback URL domain")

    allowed_lower = {d.lower() for d in allowed_domains}
    if callback_host != app_host and callback_host not in allowed_lower:
        logger.error(
            "OAuth callback domain not allowed: %s (app_host=%s, allowed=%s)",
            callback_host,
            app_host,
            allowed_domains,
        )
        raise ValueError(f"OAuth callback domain not allowed: {callback_host}")

    logger.debug("Callback domain validated: %s", callback_host)


def _apply_state_query(full_callback_parsed: urllib.parse.ParseResult, token_state: str) -> urllib.parse.ParseResult:
    """
    Pure function: Ensures the 'state' query parameter is present and set to token_state.
    If 'state' exists, it is replaced; otherwise, it is appended.
    """
    logger.debug(
        "Applying state to query. url=%s, original_query=%s, token_state=%s",
        full_callback_parsed.geturl(),
        full_callback_parsed.query,
        _mask_secret(token_state),
    )

    existing_qs = urllib.parse.parse_qsl(full_callback_parsed.query, keep_blank_values=True)
    updated_qs = []
    state_replaced = False

    for k, v in existing_qs:
        if k == "state":
            updated_qs.append(("state", token_state))
            state_replaced = True
        else:
            updated_qs.append((k, v))

    if not state_replaced:
        updated_qs.append(("state", token_state))

    new_query = urllib.parse.urlencode(updated_qs, doseq=True)
    result = full_callback_parsed._replace(query=new_query)

    logger.debug(
        "State applied. new_query=%s, result_url=%s",
        new_query,
        result.geturl(),
    )

    return result


def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Construct an OAuth callback URL by combining the application's base URL with a callback route
    and appending the 'state' token as a query parameter.

    The resulting callback URL must either:
      - Be on the same domain as the application_url, or
      - Have a hostname included in ALLOWED_CALLBACK_DOMAINS.

    Args:
        application_url: Root URL for the application.
        callback_route: Endpoint for the OAuth callback handler (absolute URL or path).
        token_state: Unique identifier to maintain the integrity of the OAuth exchange.

    Returns:
        A fully assembled OAuth callback URL as a string.

    Raises:
        ValueError: If the final callback URL's domain is not the same as application_url's domain
                    and is not included in ALLOWED_CALLBACK_DOMAINS, or if the domain cannot be
                    determined from the inputs.
    """
    if not isinstance(application_url, str) or not isinstance(callback_route, str) or not isinstance(token_state, str):
        logger.error(
            "Invalid argument types. application_url=%r, callback_route=%r, token_state=%r",
            type(application_url),
            type(callback_route),
            type(token_state),
        )
        raise TypeError("application_url, callback_route, and token_state must all be strings")

    logger.debug(
        "assemble_oauth_callback called with application_url=%s, callback_route=%s, token_state=%s",
        application_url,
        callback_route,
        _mask_secret(token_state),
    )

    # Pipeline: resolve -> validate -> apply state -> serialize
    app_parsed, full_callback_parsed = _resolve_callback(application_url, callback_route)
    _validate_callback_domain(app_parsed, full_callback_parsed, ALLOWED_CALLBACK_DOMAINS)
    final_parsed = _apply_state_query(full_callback_parsed, token_state)

    final_url = urllib.parse.urlunparse(final_parsed)
    logger.info("Assembled OAuth callback URL: %s", final_url)
    return final_url
