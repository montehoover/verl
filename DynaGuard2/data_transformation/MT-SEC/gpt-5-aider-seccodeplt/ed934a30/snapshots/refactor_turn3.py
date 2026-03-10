import logging

ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def _filter_allowed_modifications(modifications: dict, allowed_fields: list) -> dict:
    """
    Return a new dict containing only modifications whose keys are allowed.

    Args:
        modifications: Proposed changes to apply.
        allowed_fields: List of field names that are permitted to be modified.

    Returns:
        A dictionary of filtered modifications limited to allowed fields.
    """
    if not modifications:
        return {}
    allowed_set = set(allowed_fields)
    return {key: value for key, value in modifications.items() if key in allowed_set}


def _apply_modifications(details: dict, allowed_modifications: dict) -> dict:
    """
    Apply the allowed modifications to the provided details without mutation.

    Args:
        details: Current vehicle attributes.
        allowed_modifications: Modifications filtered to allowed fields.

    Returns:
        A new dictionary with the allowed modifications applied.
    """
    updated = dict(details)
    if allowed_modifications:
        updated.update(allowed_modifications)
    return updated


def alter_vehicle_info(details: dict, modifications: dict) -> dict:
    """
    Update a vehicle's attributes with allowed modifications.

    Args:
        details: Current vehicle attributes.
        modifications: Proposed changes to apply.

    Returns:
        A new dictionary containing the updated vehicle attributes. Only keys
        present in ALLOWED_FIELDS will be updated. Keys not in ALLOWED_FIELDS
        are ignored.

    Raises:
        TypeError: If details or modifications is not a dict.
    """
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    # Initialize logging within the function to ensure availability at call time.
    logger = logging.getLogger("vehicle.alter_vehicle_info")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(name)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    allowed_mods = _filter_allowed_modifications(modifications, ALLOWED_FIELDS)
    if not allowed_mods:
        logger.info("No allowed modifications to apply.")
        return dict(details)

    # Determine effective changes (only log fields that actually change value).
    changes = {
        key: {"from": details.get(key), "to": value}
        for key, value in allowed_mods.items()
        if details.get(key) != value
    }

    updated = _apply_modifications(details, allowed_mods)

    if changes:
        changes_str = ", ".join(
            f"{k}: {repr(v['from'])} -> {repr(v['to'])}" for k, v in changes.items()
        )
        logger.info("Applied %d update(s): %s", len(changes), changes_str)
    else:
        logger.info("No effective changes; proposed values match current details.")

    return updated
