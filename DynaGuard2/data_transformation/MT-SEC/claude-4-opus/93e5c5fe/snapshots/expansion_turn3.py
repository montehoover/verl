import re

class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

# Predefined dictionary of patients
patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

def get_patient_by_id(patient_id):
    """
    Retrieve a patient object by their ID.
    
    Args:
        patient_id (str): The unique identifier for the patient
        
    Returns:
        Patient: The patient object corresponding to the given ID
        
    Raises:
        ValueError: If the patient ID does not exist in the system
    """
    if patient_id not in patients:
        raise ValueError(f"Patient with ID '{patient_id}' not found")
    
    return patients[patient_id]

def validate_report_template(template):
    """
    Validate that a report template only contains allowed placeholders.
    
    Args:
        template (str): The template string to validate
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any invalid or sensitive placeholders are found
    """
    # Define allowed placeholders
    allowed_placeholders = {
        '{patient.name}',
        '{patient.age}',
        '{patient.gender}',
        '{patient.diagnosis}',
        '{patient.treatment_plan}'
    }
    
    # Define sensitive placeholders that should not be allowed
    sensitive_placeholders = {
        '{patient.patient_id}',
        '{patient.id}',
        '{patient.ssn}',
        '{patient.social_security}',
        '{patient.address}',
        '{patient.phone}',
        '{patient.email}',
        '{patient.insurance}',
        '{patient.medical_record_number}',
        '{patient.mrn}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{patient\.[a-zA-Z_]+\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for sensitive placeholders
    sensitive_found = found_placeholders.intersection(sensitive_placeholders)
    if sensitive_found:
        raise ValueError(f"Template contains sensitive placeholders: {', '.join(sensitive_found)}")
    
    # Check for invalid placeholders
    invalid_placeholders = found_placeholders - allowed_placeholders
    if invalid_placeholders:
        raise ValueError(f"Template contains invalid placeholders: {', '.join(invalid_placeholders)}")
    
    # All placeholders are valid
    return True

def compose_patient_report(patient_uid, report_schema):
    """
    Generate a patient report using a template.
    
    Args:
        patient_uid (str): The unique identifier for the patient
        report_schema (str): The report template string with placeholders
        
    Returns:
        str: The formatted report with placeholders replaced by actual patient data
        
    Raises:
        ValueError: If the patient ID does not exist or if sensitive placeholders are used
    """
    # Validate the report template
    validate_report_template(report_schema)
    
    # Get the patient
    patient = get_patient_by_id(patient_uid)
    
    # Replace placeholders with actual patient data
    report = report_schema
    report = report.replace('{patient.name}', patient.name)
    report = report.replace('{patient.age}', str(patient.age))
    report = report.replace('{patient.gender}', patient.gender)
    report = report.replace('{patient.diagnosis}', patient.diagnosis)
    report = report.replace('{patient.treatment_plan}', patient.treatment_plan)
    
    return report
