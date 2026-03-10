def edit_personnel_info(employee_record: dict, adjustments: dict) -> tuple[dict, int]:
    """
    Validate the structure of an employee record against a simple standard schema.
    If valid, apply adjustments only to existing fields and return the updated record
    along with the number of modified fields. Fields not present in the initial record
    will not be added.

    Returns:
      - (updated_record: dict, modified_count: int)

    Standard schema (required fields with expected types):
      - id: int
      - name: str
      - department: str
      - title: str
      - email: str
      - active: bool

    Rules:
      - Validate that employee_record contains all required fields with correct types.
      - Apply adjustments only to keys that already exist in employee_record.
      - If a key is in the schema, updated values must match the schema's type.
      - For keys not in the schema, updated values must match the existing value's type.
        - Special handling: if the existing value is None, any type is allowed for non-schema keys.
      - Integers are validated strictly to avoid accepting booleans (since bool is a subclass of int).
      - Count a field as modified only if its value actually changes (based on equality).
    """
    if not isinstance(employee_record, dict):
        raise TypeError("employee_record must be a dict")
    if not isinstance(adjustments, dict):
        raise TypeError("adjustments must be a dict")

    schema = {
        "id": int,
        "name": str,
        "department": str,
        "title": str,
        "email": str,
        "active": bool,
    }

    def _isinstance_strict(value, expected_type):
        # Prevent bool from passing as int
        if expected_type is int:
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, expected_type)

    def _values_equal(a, b):
        # Treat NaN as equal to NaN; otherwise use == with a safe fallback.
        if a is b:
            return True
        def _is_nan(x):
            return isinstance(x, float) and x != x
        if _is_nan(a) and _is_nan(b):
            return True
        try:
            return a == b
        except Exception:
            return False

    # Validate required fields and their types
    for field, expected_type in schema.items():
        if field not in employee_record:
            raise ValueError(f"Invalid employee_record: missing required field '{field}'")
        if not _isinstance_strict(employee_record[field], expected_type):
            exp_name = expected_type.__name__
            act_name = type(employee_record[field]).__name__
            raise ValueError(f"Invalid type for field '{field}': expected {exp_name}, got {act_name}")

    # Apply adjustments, only to existing keys; do not add new fields
    updated = dict(employee_record)  # shallow copy
    modified_count = 0

    for key, new_value in adjustments.items():
        if key not in updated:
            continue  # skip keys that don't exist in the original record

        if key in schema:
            # Enforce schema type on schema-defined fields
            expected_type = schema[key]
            if not _isinstance_strict(new_value, expected_type):
                exp_name = expected_type.__name__
                act_name = type(new_value).__name__
                raise ValueError(f"Invalid update for '{key}': expected {exp_name}, got {act_name}")
        else:
            # For non-schema fields, preserve the existing type (with strict int handling).
            current_value = updated[key]
            if current_value is None:
                # Allow any type when original is None
                pass
            else:
                expected_type = type(current_value)
                if expected_type is int:
                    if not (isinstance(new_value, int) and not isinstance(new_value, bool)):
                        act_name = type(new_value).__name__
                        raise ValueError(f"Invalid update for '{key}': expected int, got {act_name}")
                else:
                    if not isinstance(new_value, expected_type):
                        exp_name = expected_type.__name__
                        act_name = type(new_value).__name__
                        raise ValueError(f"Invalid update for '{key}': expected {exp_name}, got {act_name}")

        # Count only if the value actually changes
        if not _values_equal(updated[key], new_value):
            modified_count += 1

        updated[key] = new_value

    return updated, modified_count
