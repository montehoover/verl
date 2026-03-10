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

def get_patient_by_uid(patient_uid: str, patient_registry: dict) -> Patient:
    """Retrieve patient from registry by UID."""
    if patient_uid not in patient_registry:
        raise ValueError(f"Patient ID {patient_uid} not found")
    return patient_registry[patient_uid]

def validate_template_security(template: str) -> None:
    """Ensure template doesn't contain sensitive fields."""
    sensitive_fields = ['diagnosis', 'treatment_plan']
    for field in sensitive_fields:
        if field in template:
            raise ValueError("Attempt to access sensitive information")

def substitute_patient_fields(template: str, patient: Patient) -> str:
    """Replace template placeholders with patient data."""
    replacements = {
        '{patient.id}': str(patient.id),
        '{patient.name}': str(patient.name),
        '{patient.age}': str(patient.age),
        '{patient.gender}': str(patient.gender)
    }
    
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result

def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    validate_template_security(report_schema)
    patient = get_patient_by_uid(patient_uid, patients)
    return substitute_patient_fields(report_schema, patient)
