import logging

# Configure a module-level logger that works out of the box without interfering with global logging
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def get_product_info(prod):
    """
    Return a formatted string of product information.

    Expects a dict-like object with keys: price, description, stock, category.
    Missing keys are shown as 'N/A'.
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")

    fields = ["price", "description", "stock", "category"]
    lines = []
    for key in fields:
        val = prod.get(key, "N/A")
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


def modify_product_and_log(prod, change_data):
    """
    Update the product dictionary with the provided change_data and log each change.

    For each key in change_data:
      - If the key does not exist in prod, it is added and the addition is logged.
      - If the key exists and the value changes, the change is logged.
      - If the key exists and the value is identical, no log is produced.

    Logs are emitted using this module's logger at INFO level.
    Returns the updated product dictionary (the same object passed in).

    :param prod: dict representing the product
    :param change_data: dict of field->new_value to be applied to prod
    :return: the updated product dict
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")
    if not isinstance(change_data, dict):
        raise TypeError("change_data must be a dict")

    for key, new_val in change_data.items():
        key_exists = key in prod
        old_val = prod.get(key, None)
        if not key_exists:
            _logger.info("Added field %s: %r -> %r", key, old_val, new_val)
            prod[key] = new_val
        else:
            if old_val != new_val:
                _logger.info("Updated field %s: %r -> %r", key, old_val, new_val)
                prod[key] = new_val
            # If unchanged, do nothing

    return prod
