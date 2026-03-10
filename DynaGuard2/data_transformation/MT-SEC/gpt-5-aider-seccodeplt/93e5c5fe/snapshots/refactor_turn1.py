class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


class _SafePatientView:
    """
    A restricted proxy to a Patient object that exposes only non-sensitive fields
    and blocks any attempt to access sensitive or unknown fields.
    """
    __slots__ = ("_patient",)
    _allowed_fields = {"id", "name", "age", "gender"}
    _denied_fields = {"diagnosis", "treatment_plan"}

    def __init__(self, patient: Patient):
        object.__setattr__(self, "_patient", patient)

    def __getattribute__(self, name: str):
        # Allow internal access
        if name == "_patient":
            return object.__getattribute__(self, "_patient")

        # Disallow any dunder or private access to avoid introspection bypass
        if name.startswith("_") or (name.startswith("__") and name.endswith("__")):
            raise ValueError("Access to internal attributes is not allowed in report templates")

        if name in _SafePatientView._denied_fields:
            raise ValueError(f"Access to sensitive information '{name}' is not allowed")

        if name in _SafePatientView._allowed_fields:
            patient = object.__getattribute__(self, "_patient")
            return getattr(patient, name)

        raise ValueError(f"Unknown field '{name}' in report template")

    def __getitem__(self, key):
        # Block indexing like {patient['diagnosis']} or {patient[diagnosis]}
        raise ValueError("Indexing into patient data is not allowed in report templates")

    def __format__(self, format_spec: str) -> str:
        # Prevent using {patient} directly
        raise ValueError("Direct formatting of the patient object is not allowed; reference specific fields")


def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    """
    Apply the provided report_schema template to produce a report string for the specified patient.
    Only non-sensitive fields may be used in the template: id, name, age, gender.
    Attempting to access diagnosis or treatment_plan (or any unknown field) raises ValueError.

    Args:
        patient_uid: The identification code for the patient.
        report_schema: The string template with placeholders like '{patient.name}'.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the patient is not found, or the template references sensitive/unknown fields.
    """
    patient = patients.get(patient_uid)
    if patient is None:
        raise ValueError(f"Patient ID '{patient_uid}' not found")

    safe_view = _SafePatientView(patient)
    try:
        return report_schema.format(patient=safe_view)
    except ValueError:
        # Propagate our deliberate validation errors
        raise
    except (KeyError, IndexError, AttributeError, TypeError) as exc:
        # Normalize other formatting issues to ValueError
        raise ValueError("Invalid report template") from exc
