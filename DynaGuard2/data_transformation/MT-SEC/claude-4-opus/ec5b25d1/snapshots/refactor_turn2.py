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

def get_patient_by_code(patient_code: str) -> Patient:
    """Retrieve patient by code, raises ValueError if not found."""
    if patient_code not in patients:
        raise ValueError(f"Patient with ID {patient_code} not found")
    return patients[patient_code]

def validate_template_fields(template: str) -> None:
    """Validate that template doesn't contain sensitive fields."""
    if 'diagnosis' in template or 'treatment_plan' in template:
        raise ValueError("Access to sensitive information (diagnosis or treatment_plan) is not allowed")

def process_template(template: str, patient: Patient) -> str:
    """Replace template placeholders with patient data."""
    result = template
    result = result.replace('{patient.id}', str(patient.id))
    result = result.replace('{patient.name}', str(patient.name))
    result = result.replace('{patient.age}', str(patient.age))
    result = result.replace('{patient.gender}', str(patient.gender))
    return result

def create_person_report(patient_code: str, report_template: str) -> str:
    patient = get_patient_by_code(patient_code)
    validate_template_fields(report_template)
    return process_template(report_template, patient)
