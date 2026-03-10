from string import Formatter
from types import SimpleNamespace

# The environment provides:
# class Patient:
#     def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
#         self.id = id
#         self.name = name
#         self.age = age
#         self.gender = gender
#         self.diagnosis = diagnosis
#         self.treatment_plan = treatment_plan
#
# patients = {
#     "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
# }

def create_person_report(patient_code: str, report_template: str) -> str:
    """
    Apply a provided report template to generate a string detailing non-sensitive information
    about a patient.

    Args:
        patient_code: The identification code for the patient (must exist in patients dict).
        report_template: A string template with placeholders like '{patient.name}', '{patient.age}', etc.

    Returns:
        A string with the patient's non-sensitive information formatted as per the template.

    Raises:
        ValueError: If the patient ID is not found, or if the template attempts to access
                    sensitive information (diagnosis or treatment_plan), or if the template
                    references unsupported fields/placeholders.
    """
    # Access the patients datastore from globals without referencing an undefined name
    patients_dict = globals().get("patients")
    if not isinstance(patients_dict, dict):
        raise ValueError("Patient data store not available")

    # Validate patient existence
    try:
        patient = patients_dict[patient_code]
    except KeyError:
        raise ValueError("Patient ID not found")

    # Define which fields are sensitive and which are allowed (non-sensitive)
    sensitive_fields = {"diagnosis", "treatment_plan"}
    allowed_fields = {"id", "name", "age", "gender"}

    # Parse and validate template fields
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(report_template):
        if not field_name:
            continue  # Literal text or escaped braces

        # Disallow indexing or other complex expressions
        if "[" in field_name or "]" in field_name:
            raise ValueError("Invalid template field usage")

        # Expect placeholders to start with 'patient.'
        parts = field_name.split(".")
        if parts[0] != "patient":
            raise ValueError("Invalid template field: placeholders must start with 'patient.'")

        # Require attribute access after 'patient'
        attrs = parts[1:]
        if not attrs:
            raise ValueError("Invalid template field: must reference a patient attribute (e.g., patient.name)")

        # Validate each attribute in the chain
        for attr in attrs:
            if attr in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed")
            if attr not in allowed_fields:
                # Disallow unknown attributes and chained attributes beyond allowed ones
                raise ValueError(f"Unknown patient field: {attr}")

    # Build a safe, limited view of the patient for formatting
    safe_patient = SimpleNamespace(
        id=getattr(patient, "id", None),
        name=getattr(patient, "name", None),
        age=getattr(patient, "age", None),
        gender=getattr(patient, "gender", None),
    )

    # Perform formatting
    try:
        return report_template.format(patient=safe_patient)
    except Exception as exc:
        # Normalize any formatting errors to ValueError as per spec
        raise ValueError("Invalid report template") from exc
