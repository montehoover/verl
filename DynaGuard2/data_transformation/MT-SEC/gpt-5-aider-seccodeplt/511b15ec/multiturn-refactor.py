import logging

PERMITTED_FIELDS = ["price", "description", "stock"]

logger = logging.getLogger(__name__)


def _validate_update_inputs(prod: dict, change_data: dict) -> None:
    """
    Pure validation using guard clauses to fail fast on invalid inputs.
    Raises TypeError on invalid inputs.
    """
    # Guard: both valid
    if isinstance(prod, dict) and isinstance(change_data, dict):
        return

    # Determine which inputs are invalid and log accordingly
    prod_is_valid = isinstance(prod, dict)
    change_is_valid = isinstance(change_data, dict)

    if not prod_is_valid and not change_is_valid:
        logger.error(
            "Invalid inputs: prod type=%s, change_data type=%s",
            type(prod).__name__,
            type(change_data).__name__,
        )
        raise TypeError("prod and change_data must be dicts")

    if not prod_is_valid:
        logger.error("Invalid input: prod type=%s", type(prod).__name__)
        raise TypeError("prod must be a dict")

    logger.error("Invalid input: change_data type=%s", type(change_data).__name__)
    raise TypeError("change_data must be a dict")


def _compute_allowed_updates(change_data: dict, permitted_fields: list) -> dict:
    """
    Pure function that computes which updates are permitted without mutating inputs.
    Returns a new dict containing only permitted field updates.
    """
    return {key: value for key, value in change_data.items() if key in permitted_fields}


def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Update permitted fields of a product dict using values from change_data.

    Only keys listed in PERMITTED_FIELDS will be updated.
    Returns a dictionary of the fields that were updated with their new values.
    """
    _validate_update_inputs(prod, change_data)

    # Guard: no incoming changes
    if not change_data:
        logger.info("No change_data provided; nothing to update.")
        return {}

    logger.debug("Incoming change_data: %s", change_data)

    applied_changes = _compute_allowed_updates(change_data, PERMITTED_FIELDS)
    if not applied_changes:
        logger.info("No permitted fields to update. Permitted=%s, provided=%s", PERMITTED_FIELDS, list(change_data.keys()))
        return {}

    before_snapshot = {k: prod.get(k) for k in applied_changes.keys()}
    logger.debug("Before update snapshot (permitted keys): %s", before_snapshot)
    logger.debug("Allowed updates computed: %s", applied_changes)

    # Apply computed changes to the product (side-effect occurs here by design)
    for key, value in applied_changes.items():
        prod[key] = value

    after_snapshot = {k: prod.get(k) for k in applied_changes.keys()}
    logger.info(
        "Applied product updates. Changes=%s | Before=%s | After=%s",
        applied_changes,
        before_snapshot,
        after_snapshot,
    )

    return applied_changes
