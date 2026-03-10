from typing import Any, Dict, Tuple, List
from copy import deepcopy


def _is_number_like(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
            return True
        except ValueError:
            return False
    return False


def _coerce_number(value: Any) -> Tuple[bool, Any, str]:
    # Accept int/float or numeric strings. Returns float.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, float(value), ""
    if isinstance(value, str):
        try:
            return True, float(value.strip()), ""
        except ValueError:
            pass
    return False, None, "expected a number"


def _coerce_integer(value: Any) -> Tuple[bool, Any, str]:
    # Accept ints, floats that are integral, or numeric strings representing an int.
    if isinstance(value, bool):
        return False, None, "expected an integer"
    if isinstance(value, int):
        return True, value, ""
    if isinstance(value, float) and value.is_integer():
        return True, int(value), ""
    if isinstance(value, str):
        v = value.strip()
        try:
            iv = int(v)
            # Ensure string is an integer representation (no decimals)
            if str(iv) == v or v.startswith("+") and str(iv) == v[1:]:
                return True, iv, ""
        except ValueError:
            pass
    return False, None, "expected an integer"


def _coerce_bool(value: Any) -> Tuple[bool, Any, str]:
    if isinstance(value, bool):
        return True, value, ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, bool(value), ""
    if isinstance(value, str):
        v = value.strip().lower()
        truthy = {"true", "1", "yes", "y", "on"}
        falsy = {"false", "0", "no", "n", "off"}
        if v in truthy:
            return True, True, ""
        if v in falsy:
            return True, False, ""
    return False, None, "expected a boolean"


def _coerce_string(value: Any) -> Tuple[bool, Any, str]:
    if isinstance(value, str):
        return True, value, ""
    return False, None, "expected a string"


def _coerce_list_of_str(value: Any) -> Tuple[bool, Any, str]:
    # Accept list of strings or a single comma-separated string.
    if isinstance(value, list):
        if all(isinstance(x, str) for x in value):
            return True, value, ""
        return False, None, "expected a list of strings"
    if isinstance(value, str):
        # Split by comma and trim
        items = [item.strip() for item in value.split(",")]
        return True, items, ""
    return False, None, "expected a list of strings"


def _coerce_dict(value: Any) -> Tuple[bool, Any, str]:
    if isinstance(value, dict):
        return True, value, ""
    return False, None, "expected an object/dict"


def _infer_schema_for_field(field: str, current_value: Any) -> str:
    """
    Return a schema type identifier for the given field:
    'integer', 'number', 'boolean', 'string', 'list[str]', 'dict', or 'any'
    """
    name = field.lower()

    # Inference by existing value type
    if isinstance(current_value, bool):
        return "boolean"
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return "integer"
    if isinstance(current_value, float):
        return "number"
    if isinstance(current_value, str):
        return "string"
    if isinstance(current_value, list):
        # Assume list of strings for common fields; otherwise 'list[str]' as default
        return "list[str]"
    if isinstance(current_value, dict):
        return "dict"
    if current_value is None:
        # Heuristics by field name if value is None
        number_like_names = {"price", "cost", "amount", "weight", "length", "width", "height", "msrp"}
        integer_like_names = {"qty", "quantity", "stock", "inventory", "count"}
        boolean_like_names = {"active", "is_active", "available", "in_stock", "published", "visible"}
        list_like_names = {"tags", "categories", "labels", "keywords"}
        dict_like_names = {"dimensions", "attributes", "metadata", "options"}

        if name in integer_like_names:
            return "integer"
        if name in number_like_names:
            return "number"
        if name in boolean_like_names:
            return "boolean"
        if name in list_like_names:
            return "list[str]"
        if name in dict_like_names:
            return "dict"
        # Strings for typical text fields
        string_like_names = {"name", "title", "description", "sku", "slug", "brand", "model", "id"}
        if name in string_like_names:
            return "string"

    # Fallback
    return "any"


def _coerce_value_to_schema(schema: str, value: Any) -> Tuple[bool, Any, str]:
    if schema == "integer":
        return _coerce_integer(value)
    if schema == "number":
        return _coerce_number(value)
    if schema == "boolean":
        return _coerce_bool(value)
    if schema == "string":
        return _coerce_string(value)
    if schema == "list[str]":
        return _coerce_list_of_str(value)
    if schema == "dict":
        return _coerce_dict(value)
    # 'any' accepts any incoming value
    return True, value, ""


def update_product_info(prod: Dict[str, Any], change_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and apply changes to a product dict.

    Arguments:
      - prod: dict - existing product with fields like price, description, etc.
      - change_data: dict - incoming values to validate and apply

    Behavior:
      - Only fields present in `prod` are considered; unknown fields are rejected.
      - Each value in `change_data` is validated against an inferred schema
        (based on the current value type or common field-name heuristics).
      - Numeric fields accept numbers or numeric strings; boolean fields accept common textual booleans.
      - Strings must be strings. Lists of strings accept either a list[str] or a comma-separated string.
      - On success, valid changes are applied to a copy of `prod`.

    Returns:
      A dictionary with:
        - product: the updated product dictionary with validated inputs applied
        - applied: dict of field -> coerced value that were applied
        - errors: dict of field -> error message for rejected inputs
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")
    if not isinstance(change_data, dict):
        raise TypeError("change_data must be a dict")

    updated = deepcopy(prod)
    applied: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for field, new_value in change_data.items():
        if field not in prod:
            errors[field] = "unknown field"
            continue

        if new_value is None:
            errors[field] = "value cannot be null"
            continue

        schema = _infer_schema_for_field(field, prod.get(field))
        ok, coerced, err = _coerce_value_to_schema(schema, new_value)
        if ok:
            updated[field] = coerced
            applied[field] = coerced
        else:
            errors[field] = err

    return {
        "product": updated,
        "applied": applied,
        "errors": errors,
    }
