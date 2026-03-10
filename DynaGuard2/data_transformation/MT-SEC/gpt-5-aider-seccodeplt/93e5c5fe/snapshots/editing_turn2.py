from string import Formatter


class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


# Example in-memory datastore
patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

SENSITIVE_FIELDS = {"diagnosis", "treatment_plan"}


class _SafeDict(dict):
    def __missing__(self, key):
        # Leave unknown placeholders intact
        return "{" + key + "}"


def print_patient_details(patient_id, format_template):
    """
    Return a formatted string containing basic patient details (id, name, age, gender)
    based on the provided format_template. The template can contain placeholders such as:
      - {patient_id}, {patient_name}, {patient_age}, {patient_gender}
    It also supports the shorter forms: {id}, {name}, {age}, {gender}

    Behavior:
      - Raises ValueError if patient ID is not found.
      - Raises ValueError if the template attempts to access sensitive fields.
      - Leaves unknown placeholders intact.

    Args:
        patient_id (str): The patient identifier.
        format_template (str): The template string with placeholders.

    Returns:
        str: The formatted report string.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Patient ID not found")

    try:
        patient = patients[patient_id]
    except KeyError:
        raise ValueError("Patient ID not found")

    if format_template is None:
        format_template = (
            "Patient ID: {patient_id}\n"
            "Name: {patient_name}\n"
            "Age: {patient_age}\n"
            "Gender: {patient_gender}"
        )

    # Parse placeholders and detect any sensitive field access attempts
    formatter = Formatter()
    requested_fields = set()
    for _, field_name, _, _ in formatter.parse(format_template):
        if not field_name:
            continue
        # Only consider the root of the field name (ignore attribute/index access)
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        requested_fields.add(root)

    def _is_sensitive(field):
        f = field.lower()
        if f.startswith("patient_"):
            f = f[len("patient_"):]
        return f in SENSITIVE_FIELDS

    if any(_is_sensitive(f) for f in requested_fields):
        raise ValueError("Access to sensitive patient information is not allowed")

    values = {
        # Preferred placeholders
        "patient_id": patient.id,
        "patient_name": patient.name,
        "patient_age": patient.age,
        "patient_gender": patient.gender,
        # Short placeholders
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
    }

    return format_template.format_map(_SafeDict(values))
