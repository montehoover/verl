from string import Formatter
from types import SimpleNamespace


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

# Define which patient fields are allowed to be referenced in templates
BASIC_ALLOWED_PATIENT_FIELDS = frozenset({"id", "name", "age", "gender"})


def render_patient_template(report_template: str, patient: Patient) -> str:
    """
    Pure function that renders a report template with a patient's basic information.
    Only exposes basic, non-restricted fields defined in BASIC_ALLOWED_PATIENT_FIELDS.

    Args:
        report_template: The string template containing placeholders like '{patient.name}'.
        patient: The Patient instance whose data will be injected.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the template attempts to access restricted medical data
                    or uses invalid placeholders.
    """
    allowed_attrs = set(BASIC_ALLOWED_PATIENT_FIELDS)
    allowed_fields = {f"patient.{attr}" for attr in allowed_attrs}

    # Validate template placeholders
    formatter = Formatter()
    for _, field_name, _format_spec, _conversion in formatter.parse(report_template):
        if field_name is None:
            continue

        # Reject any field not explicitly allowed
        if field_name not in allowed_fields:
            if field_name.startswith("patient."):
                # Attempt to access a restricted or unknown attribute on patient
                attr = field_name.split(".", 1)[1]
                if attr not in allowed_attrs:
                    raise ValueError("Attempted access to restricted patient data")
            else:
                # Attempt to access non-patient fields is invalid
                raise ValueError("Invalid report template placeholders")

    # Build a safe object exposing only allowed attributes
    safe_patient = SimpleNamespace(**{attr: getattr(patient, attr) for attr in allowed_attrs})

    try:
        return report_template.format(patient=safe_patient)
    except Exception as exc:
        # Normalize any formatting issues to ValueError as per spec
        raise ValueError("Invalid report template") from exc


def create_patient_report(id: str, report_template: str) -> str:
    """
    Generates a patient report by filling a template with basic patient information.

    Args:
        id: The unique identifier of the patient within the system.
        report_template: The string template containing placeholders like '{patient.name}'.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid or if the template attempts to access
                    restricted medical data.
    """
    if not isinstance(id, str) or not id:
        raise ValueError("Invalid patient ID")

    if id not in patients:
        raise ValueError("Invalid patient ID")

    patient = patients[id]
    return render_patient_template(report_template, patient)
