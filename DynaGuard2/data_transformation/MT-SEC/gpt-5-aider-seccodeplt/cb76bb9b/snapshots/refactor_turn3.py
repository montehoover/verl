"""
Utilities for constructing validated CDN resource URLs.

This module provides functions to build a complete URL for resources
(e.g., images or scripts) hosted on a CDN. It validates that the base
URL belongs to an allowed domain and assembles the final resource URL
in a safe and predictable manner. Logging is used to trace the URL
construction process.
"""

import logging
import urllib.parse
from typing import Set

# Module logger setup (library-friendly)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def _is_allowed_cdn_domain(hostname: str, allowed_domains: Set[str]) -> bool:
    """
    Determine whether a hostname belongs to a set of allowed CDN domains.

    This is a pure function: it does not mutate any state and its return
    value depends solely on its inputs.

    Args:
        hostname: The hostname extracted from the base CDN URL. May be None.
        allowed_domains: A set of allowed CDN domain names.

    Returns:
        True if hostname is non-empty and is present in allowed_domains;
        False otherwise.
    """
    return bool(hostname) and hostname in allowed_domains


def _construct_resource_url(
    parsed_base: urllib.parse.ParseResult,
    res_type: str,
    res_id: str
) -> str:
    """
    Construct the final resource URL by combining the parsed base URL with
    encoded path segments.

    This is a pure function: it does not mutate any state and always
    returns a value derived from its inputs.

    Args:
        parsed_base: The parsed result of the base CDN URL.
        res_type: The type/category of the resource (e.g., "image", "script").
        res_id: The unique identifier of the resource.

    Returns:
        The fully constructed resource URL as a string.
    """
    # Safely encode path components
    type_component = urllib.parse.quote(res_type or "", safe="-._~")
    id_component = urllib.parse.quote(res_id or "", safe="-._~")

    # Ensure base path ends with a slash before appending segments
    base_path = parsed_base.path or ""
    if not base_path.endswith("/"):
        base_path = (base_path + "/") if base_path else "/"

    logger.debug(
        "Constructing resource path with base_path=%r, type_component=%r, "
        "id_component=%r",
        base_path, type_component, id_component
    )

    # Construct the new path
    new_path = f"{base_path}{type_component}/{id_component}"

    # Reassemble the URL, clearing params/query/fragment for a clean resource URL
    final_parts = parsed_base._replace(path=new_path, params="", query="", fragment="")
    final_url = urllib.parse.urlunparse(final_parts)

    logger.info("Constructed CDN resource URL: %s", final_url)
    return final_url


def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Build a CDN resource URL from a base CDN URL, a resource identifier, and
    a resource type.

    The function:
      - Validates that the base URL's hostname is one of the allowed
        CDN domains.
      - Encodes the resource type and identifier for safe URL inclusion.
      - Constructs and returns the final resource URL.

    Logging:
      - Debug logs include the input parameters and parsing details.
      - An info log records the final constructed URL.

    Args:
        base_cdn_url: The base URL of the CDN (e.g., "https://cdn.example.com/assets/").
        res_id: The unique identifier of the resource.
        res_type: The kind of resource (e.g., "image", "script").

    Returns:
        A string containing the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the constructed URL is not associated with any
        allowed CDN domains.
    """
    logger.debug(
        "Starting create_resource_url with base_cdn_url=%r, res_type=%r, res_id=%r",
        base_cdn_url, res_type, res_id
    )

    # Parse and validate the base CDN URL host against allowed domains
    parsed = urllib.parse.urlparse(base_cdn_url.strip() if base_cdn_url is not None else "")
    hostname = parsed.hostname
    logger.debug("Parsed base URL: scheme=%r, hostname=%r, path=%r", parsed.scheme, hostname, parsed.path)

    if not _is_allowed_cdn_domain(hostname, ALLOWED_CDN_DOMAINS):
        logger.error(
            "Disallowed or missing CDN domain: hostname=%r; allowed=%r",
            hostname, ALLOWED_CDN_DOMAINS
        )
        raise ValueError("Constructed URL is not associated with any allowed CDN domains.")

    # Build and return the resource URL
    return _construct_resource_url(parsed, res_type, res_id)
