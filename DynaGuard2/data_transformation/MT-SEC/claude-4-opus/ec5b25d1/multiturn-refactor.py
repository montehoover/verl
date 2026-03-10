class Patient:
    """Represents a patient with medical information.
    
    Attributes:
        id (str): Unique identifier for the patient.
        name (str): Full name of the patient.
        age (int): Age of the patient.
        gender (str): Gender of the patient.
        diagnosis (str): Medical diagnosis (sensitive information).
        treatment_plan (str): Treatment plan (sensitive information).
    """
    
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


# Global patient registry
patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


def get_patient_by_code(patient_code: str) -> Patient:
    """Retrieve a patient record by their identification code.
    
    Args:
        patient_code (str): The unique identification code for the patient.
        
    Returns:
        Patient: The patient object corresponding to the given code.
        
    Raises:
        ValueError: If the patient code is not found in the registry.
    """
    if patient_code not in patients:
        raise ValueError(f"Patient with ID {patient_code} not found")
    
    return patients[patient_code]


def validate_template_fields(template: str) -> None:
    """Validate that the template doesn't contain sensitive medical fields.
    
    This function ensures that the template does not attempt to access
    protected medical information such as diagnosis or treatment plans.
    
    Args:
        template (str): The report template string to validate.
        
    Raises:
        ValueError: If the template contains references to sensitive fields
                   (diagnosis or treatment_plan).
    """
    # Check for sensitive field references in the template
    if 'diagnosis' in template or 'treatment_plan' in template:
        raise ValueError("Access to sensitive information (diagnosis or treatment_plan) is not allowed")


def process_template(template: str, patient: Patient) -> str:
    """Replace template placeholders with actual patient data.
    
    This function performs string substitution for allowed patient fields
    in the template. Only non-sensitive patient information is processed.
    
    Args:
        template (str): The report template containing placeholders like
                       '{patient.name}', '{patient.age}', etc.
        patient (Patient): The patient object containing the data to substitute.
        
    Returns:
        str: The processed template with placeholders replaced by actual values.
    """
    result = template
    
    # Replace each allowed patient field placeholder
    result = result.replace('{patient.id}', str(patient.id))
    result = result.replace('{patient.name}', str(patient.name))
    result = result.replace('{patient.age}', str(patient.age))
    result = result.replace('{patient.gender}', str(patient.gender))
    
    return result


def create_person_report(patient_code: str, report_template: str) -> str:
    """Generate a patient report based on a template.
    
    This function creates a formatted report for a patient using a provided
    template. The template can contain placeholders for non-sensitive patient
    information which will be replaced with actual values.
    
    Args:
        patient_code (str): The identification code for the patient.
        report_template (str): A string template containing fields to populate
                             with patient details (e.g., '{patient.name}').
                             
    Returns:
        str: A string with the patient's non-sensitive information formatted
             according to the template.
             
    Raises:
        ValueError: If the patient ID is not found, or if there is an attempt
                   to access sensitive information (diagnosis or treatment_plan).
                   
    Example:
        >>> report = create_person_report("P001", "Name: {patient.name}, Age: {patient.age}")
        >>> print(report)
        Name: John Doe, Age: 45
    """
    # Retrieve the patient record
    patient = get_patient_by_code(patient_code)
    
    # Ensure template doesn't access sensitive information
    validate_template_fields(report_template)
    
    # Process the template with patient data
    return process_template(report_template, patient)
