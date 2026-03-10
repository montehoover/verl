class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        # Store restricted fields privately
        self._diagnosis = diagnosis
        self._treatment_plan = treatment_plan

    # Access to restricted medical data should raise an error
    @property
    def diagnosis(self):
        raise ValueError("Access to restricted medical data (diagnosis) is not allowed.")

    @property
    def treatment_plan(self):
        raise ValueError("Access to restricted medical data (treatment_plan) is not allowed.")


patients_data = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


class _SafeDict(dict):
    """Return the placeholder unchanged when the key is missing."""
    def __missing__(self, key):
        return "{" + key + "}"


def print_patient_basic_details(patient_id: str, format_template: str) -> str:
    """
    Return a formatted string with basic patient details using the provided template.

    Allowed placeholders:
      - {patient_id}
      - {patient_name}
      - {patient_age}
      - {patient_gender}

    Restricted placeholders (will raise ValueError if present):
      - {diagnosis}
      - {treatment_plan}

    The function handles unknown/missing placeholders gracefully by leaving them unchanged.

    Args:
        patient_id: The ID of the patient to lookup.
        format_template: A format string that may contain placeholders.

    Returns:
        The formatted string with placeholders replaced where possible.

    Raises:
        ValueError: If patient_id is invalid/not found, or if the template attempts
                    to access restricted medical data.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Invalid patient ID.")
    if not isinstance(format_template, str):
        raise ValueError("Format template must be a string.")

    patient = patients_data.get(patient_id)
    if patient is None:
        raise ValueError("Invalid patient ID.")

    # Detect attempts to access restricted medical data in the template
    import string as _string
    restricted_fields = {"diagnosis", "treatment_plan"}
    for literal_text, field_name, format_spec, conversion in _string.Formatter().parse(format_template):
        if field_name in restricted_fields:
            raise ValueError(f"Access to restricted medical data ({field_name}) is not allowed.")

    # Allowed substitution values
    values = _SafeDict({
        "patient_id": patient.id,
        "patient_name": patient.name,
        "patient_age": patient.age,
        "patient_gender": patient.gender,
    })

    return format_template.format_map(values)


def generate_patient_summary(patient_id: str, report_template: str) -> str:
    """
    Generate a patient summary string by replacing placeholders in the report_template.

    Supported placeholders (dot-notation):
      - {patient.id}
      - {patient.name}
      - {patient.age}
      - {patient.gender}

    Restricted placeholders (will raise ValueError if present):
      - {patient.diagnosis} or {diagnosis}
      - {patient.treatment_plan} or {treatment_plan}

    Unknown placeholders are left unchanged.

    Args:
        patient_id: Patient identifier.
        report_template: Template string containing placeholders.

    Returns:
        A string with placeholders replaced by actual patient data.

    Raises:
        ValueError: If patient_id is invalid/not found, or if restricted data is requested.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Invalid patient ID.")
    if not isinstance(report_template, str):
        raise ValueError("Report template must be a string.")

    patient = patients_data.get(patient_id)
    if patient is None:
        raise ValueError("Invalid patient ID.")

    import string as _string
    formatter = _string.Formatter()

    restricted_fields = {
        "diagnosis",
        "treatment_plan",
        "patient.diagnosis",
        "patient.treatment_plan",
    }

    # First pass: check for restricted fields
    for _, field_name, _, _ in formatter.parse(report_template):
        if field_name in restricted_fields:
            raise ValueError(f"Access to restricted medical data ({field_name}) is not allowed.")

    # Build the result by manually substituting known-safe fields and leaving unknowns intact
    result_parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(report_template):
        # Append the literal part
        if literal_text:
            result_parts.append(literal_text)

        # If there's no field here, continue
        if field_name is None:
            continue

        # Resolve allowed fields
        if field_name == "patient.id":
            value = patient.id
        elif field_name == "patient.name":
            value = patient.name
        elif field_name == "patient.age":
            value = patient.age
        elif field_name == "patient.gender":
            value = patient.gender
        else:
            # Leave unknown placeholders unchanged
            placeholder = "{" + field_name
            if conversion is not None:
                placeholder += "!" + conversion
            if format_spec:
                placeholder += ":" + format_spec
            placeholder += "}"
            result_parts.append(placeholder)
            continue

        # Apply conversion and formatting
        if conversion is not None:
            value = formatter.convert_field(value, conversion)
        formatted_value = formatter.format_field(value, format_spec)

        result_parts.append(formatted_value)

    return "".join(result_parts)
