import string
from typing import Set


# Non-sensitive and disallowed (sensitive) patient fields
NON_SENSITIVE_FIELDS: Set[str] = {"id", "name", "age", "gender"}
DISALLOWED_FIELDS: Set[str] = {"diagnosis", "treatment_plan"}


class _AllowedPatientView:
    """
    Read-only, restricted view over a Patient object that only exposes
    non-sensitive fields defined in NON_SENSITIVE_FIELDS.
    """
    __slots__ = ("_p",)

    def __init__(self, patient_obj):
        self._p = patient_obj

    def __getattr__(self, name: str):
        if name in NON_SENSITIVE_FIELDS:
            return getattr(self._p, name)
        # Prevent any access to sensitive or unknown attributes
        raise AttributeError(f"Access to field '{name}' is not allowed")


def _resolve_patients_registry():
    """
    Resolve the patients registry without directly referencing an undefined
    name to satisfy static analysis. It tries module globals first, then
    falls back to __main__.
    """
    reg = globals().get("patients")
    if reg is None:
        try:
            import sys  # local import to avoid unnecessary global scope pollution
            main_mod = sys.modules.get("__main__")
        except Exception:
            main_mod = None
        reg = getattr(main_mod, "patients", None) if main_mod is not None else None

    if reg is None:
        raise ValueError("Patients registry is not available")
    return reg


def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    """
    Apply a report template to a patient's non-sensitive information.

    Args:
        patient_uid: The identification code for the patient.
        report_schema: A string template containing placeholders like
                       '{patient.name}' or '{patient.age}'.

    Returns:
        A formatted string with the patient's non-sensitive information.

    Raises:
        ValueError: If the patient ID is not found, or if the template attempts
                    to access sensitive information (diagnosis or treatment_plan),
                    or attempts to access unsupported fields.
    """
    # Validate patient existence
    registry = _resolve_patients_registry()
    try:
        patient = registry[patient_uid]
    except KeyError:
        raise ValueError(f"Patient ID '{patient_uid}' not found") from None

    # Pre-validate all fields referenced in the template to ensure only allowed access
    formatter = string.Formatter()
    for _literal, field_name, _format_spec, _conversion in formatter.parse(report_schema):
        if not field_name:
            continue

        # Only allow fields of the form 'patient.<attr>'
        if field_name == "patient":
            raise ValueError("Direct use of '{patient}' is not allowed; specify a field like '{patient.name}'")

        if not field_name.startswith("patient."):
            raise ValueError(f"Unsupported placeholder '{{{field_name}}}'. Use '{{patient.<field>}}'")

        parts = field_name.split(".")
        if len(parts) != 2:
            # Disallow nested attributes or deeper traversal like patient.a.b or any other tricks
            raise ValueError(f"Unsupported nested placeholder '{{{field_name}}}'. Use '{{patient.<field>}}'")

        attr = parts[1]

        # Disallow any indexing or suspicious constructs in the attribute portion
        if any(ch in attr for ch in "[]"):
            raise ValueError(f"Indexing is not allowed in placeholders: '{{{field_name}}}'")

        # Enforce sensitive field restrictions
        if attr in DISALLOWED_FIELDS:
            raise ValueError(f"Access to sensitive field '{attr}' is not permitted")

        # Enforce that only known non-sensitive fields are used
        if attr not in NON_SENSITIVE_FIELDS:
            raise ValueError(f"Unknown or disallowed field '{attr}'. Allowed fields: {sorted(NON_SENSITIVE_FIELDS)}")

    # Use a restricted proxy to prevent accidental access during formatting
    safe_patient = _AllowedPatientView(patient)

    try:
        result = report_schema.format(patient=safe_patient)
    except Exception as exc:
        # Normalize any formatting-related error to ValueError for the API contract
        raise ValueError(f"Failed to compose report: {exc}") from None

    return result
