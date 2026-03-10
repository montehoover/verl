from typing import Dict
import string


class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


patients: Dict[str, Patient] = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

_RESTRICTED_ATTRS = {"diagnosis", "treatment_plan"}


def _safe_get(patient: Patient, attr: str):
    if attr in _RESTRICTED_ATTRS:
        raise ValueError("Access to restricted medical data is not allowed.")
    return getattr(patient, attr)


class _DefaultStrDict(dict):
    def __missing__(self, key):
        # Leave unknown placeholders unchanged
        return "{" + key + "}"


def print_patient_details(patient_id: str, format_template: str) -> str:
    """
    Return a formatted string with basic patient details substituted into the provided template.

    Supported placeholders:
      - {patient_id}
      - {patient_name}
      - {patient_age}
      - {patient_gender}

    Any unknown placeholders are left unchanged. Attempting to access restricted data
    (e.g., {diagnosis} or {patient_treatment_plan}) raises ValueError.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Invalid patient ID.")

    try:
        patient = patients[patient_id]
    except KeyError:
        raise ValueError("Invalid patient ID.")

    if not isinstance(format_template, str):
        raise ValueError("format_template must be a string.")

    # Detect attempts to access restricted data via placeholder names
    restricted_placeholders = {
        "diagnosis",
        "treatment_plan",
        "patient_diagnosis",
        "patient_treatment_plan",
    }
    for _, field_name, _, _ in string.Formatter().parse(format_template):
        if field_name and field_name in restricted_placeholders:
            raise ValueError("Access to restricted medical data is not allowed.")

    # Build safe substitution mapping
    mapping = _DefaultStrDict(
        {
            "patient_id": _safe_get(patient, "id"),
            "patient_name": _safe_get(patient, "name"),
            "patient_age": _safe_get(patient, "age"),
            "patient_gender": _safe_get(patient, "gender"),
        }
    )

    return format_template.format_map(mapping)


def create_patient_report(id: str, report_template: str) -> str:
    """
    Return a report string by substituting placeholders like:
      - {patient.name}
      - {patient.age}
      - {patient.gender}
      - {patient.id}

    Rules:
      - Raises ValueError if id is invalid.
      - Raises ValueError if template attempts to access restricted attributes
        (diagnosis, treatment_plan).
      - Placeholders not starting with 'patient.' are left unchanged.
    """
    if not isinstance(id, str) or not id.strip():
        raise ValueError("Invalid patient ID.")

    try:
        patient = patients[id]
    except KeyError:
        raise ValueError("Invalid patient ID.")

    if not isinstance(report_template, str):
        raise ValueError("report_template must be a string.")

    formatter = string.Formatter()
    result_parts = []

    def _reconstruct_placeholder(field_name: str, conversion, format_spec: str) -> str:
        out = "{" + field_name
        if conversion:
            out += "!" + conversion
        if format_spec:
            out += ":" + format_spec
        out += "}"
        return out

    for literal_text, field_name, format_spec, conversion in formatter.parse(report_template):
        # Append literal part
        if literal_text:
            result_parts.append(literal_text)

        # If there's no field, continue
        if field_name is None:
            continue

        # Only handle placeholders that start with 'patient.'
        if field_name.startswith("patient."):
            parts = field_name.split(".")

            # Must be exactly 'patient.<attr>'. Deeper nesting is not supported; leave unchanged.
            if len(parts) != 2:
                # If an immediate attribute is restricted, still raise.
                if len(parts) >= 2 and parts[1] in _RESTRICTED_ATTRS:
                    raise ValueError("Access to restricted medical data is not allowed.")
                result_parts.append(_reconstruct_placeholder(field_name, conversion, format_spec))
                continue

            attr = parts[1]
            if attr in _RESTRICTED_ATTRS:
                raise ValueError("Access to restricted medical data is not allowed.")

            if not hasattr(patient, attr):
                # Unknown attribute: leave placeholder unchanged
                result_parts.append(_reconstruct_placeholder(field_name, conversion, format_spec))
                continue

            value = getattr(patient, attr)

            # Apply conversion first (!r, !s, !a), mirroring str.format behavior.
            if conversion == "r":
                value = repr(value)
            elif conversion == "s":
                value = str(value)
            elif conversion == "a":
                value = ascii(value)
            elif conversion is not None:
                # Unknown conversion: leave placeholder unchanged
                result_parts.append(_reconstruct_placeholder(field_name, conversion, format_spec))
                continue

            # Apply format specification if provided
            formatted = format(value, format_spec) if format_spec else f"{value}"
            result_parts.append(f"{formatted}")
        else:
            # Leave non-patient placeholders unchanged
            result_parts.append(_reconstruct_placeholder(field_name, conversion, format_spec))

    return "".join(result_parts)
