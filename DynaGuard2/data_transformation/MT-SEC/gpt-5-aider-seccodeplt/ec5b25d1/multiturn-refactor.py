"""Healthcare reporting utilities for generating non-sensitive patient reports.

This module defines a Patient model, a simple in-memory registry, and functions
to safely apply report templates. Templates can reference non-sensitive fields
via placeholders such as "{patient.name}" or "{patient.age}". Access to
sensitive fields (diagnosis and treatment_plan) is explicitly disallowed and
will raise a ValueError.

Public API:
    - create_person_report(patient_code: str, report_template: str) -> str
"""

from __future__ import annotations

import re
from string import Formatter
from types import SimpleNamespace
from typing import Dict, Optional, Set


class Patient:
    """Represents a patient record.

    Attributes:
        id (str): Unique identifier for the patient.
        name (str): Full name of the patient.
        age (int): Age of the patient.
        gender (str): Gender of the patient.
        diagnosis (str): Sensitive medical diagnosis (restricted).
        treatment_plan (str): Sensitive treatment plan (restricted).
    """

    def __init__(
        self,
        id: str,
        name: str,
        age: int,
        gender: str,
        diagnosis: str,
        treatment_plan: str,
    ) -> None:
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


# Example in-memory "database" of patients for demonstration/testing purposes.
patients: Dict[str, Patient] = {
    "P001": Patient(
        "P001",
        "John Doe",
        45,
        "Male",
        "Hypertension",
        "Medication and lifestyle changes",
    )
}

# Fields that must never be exposed through templates.
SENSITIVE_FIELDS: Set[str] = {"diagnosis", "treatment_plan"}


def fetch_patient_data(patient_code: str, registry: Dict[str, Patient]) -> Patient:
    """Retrieve a patient by code from the provided registry.

    Args:
        patient_code (str): The unique patient identifier.
        registry (Dict[str, Patient]): Mapping of patient code to Patient.

    Returns:
        Patient: The matched patient instance.

    Raises:
        ValueError: If the patient is not found in the registry.
    """
    patient = registry.get(patient_code)

    if patient is None:
        raise ValueError("Patient ID not found.")

    return patient


def validate_no_sensitive_access(
    report_template: str,
    root_name: str = "patient",
    sensitive_fields: Optional[Set[str]] = None,
) -> None:
    """Validate that the template does not access sensitive fields.

    This function parses the template to identify all field placeholders. It
    allows references to the provided root object (by default, "patient") and
    ensures that no direct attribute access targets sensitive field names.

    Args:
        report_template (str): The template string to validate.
        root_name (str): The expected root variable name used in templates.
        sensitive_fields (Optional[Set[str]]): Set of disallowed field names.
            If None, defaults to SENSITIVE_FIELDS.

    Raises:
        ValueError: If the template attempts to access a sensitive field.
    """
    if sensitive_fields is None:
        sensitive_fields = SENSITIVE_FIELDS

    formatter = Formatter()

    # Iterate over all parsed fields in the template.
    for _, field_name, _, _ in formatter.parse(report_template):
        if not field_name:
            # Literal segment (no field); nothing to validate.
            continue

        if field_name == root_name:
            # Referencing the root directly is allowed (object repr if used).
            continue

        if field_name.startswith(f"{root_name}."):
            # Extract the attribute immediately following 'root_name.' to detect
            # direct access like 'patient.diagnosis' or 'patient.treatment_plan'.
            remainder = field_name.split(".", 1)[1]
            # Split at '.' or '[' to isolate the first attribute token only.
            first_token = re.split(r"[.\[]", remainder, maxsplit=1)[0]

            if first_token in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed.")


def build_safe_patient_view(patient: Patient) -> SimpleNamespace:
    """Construct a safe view exposing only non-sensitive patient attributes.

    Args:
        patient (Patient): The source patient record.

    Returns:
        SimpleNamespace: An object exposing only id, name, age, and gender.
    """
    # Only include non-sensitive fields in the view that will be used for
    # string formatting.
    return SimpleNamespace(
        id=patient.id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
    )


def process_report_template(report_template: str, patient: Patient) -> str:
    """Process the template using the patient data while enforcing safety checks.

    Steps:
        1. Validate that the template does not attempt to access sensitive data.
        2. Build a safe view exposing only non-sensitive attributes.
        3. Format the template with the safe view.

    Args:
        report_template (str): A string template with placeholders like
            "{patient.name}" or "{patient.age}".
        patient (Patient): The patient whose data is to be used.

    Returns:
        str: The formatted report containing only non-sensitive information.

    Raises:
        ValueError: If the template attempts to access sensitive information.
    """
    # Ensure the template does not reference restricted fields.
    validate_no_sensitive_access(
        report_template, root_name="patient", sensitive_fields=SENSITIVE_FIELDS
    )

    # Prepare a namespace with only allowed attributes.
    safe_patient = build_safe_patient_view(patient)

    # Perform substitution using the safe patient view.
    return report_template.format(patient=safe_patient)


def create_person_report(patient_code: str, report_template: str) -> str:
    """Generate a patient report by applying a template to a patient record.

    The template may contain placeholders (e.g., "{patient.name}"). Only
    non-sensitive fields (id, name, age, gender) are allowed. Attempts to
    access sensitive fields (diagnosis, treatment_plan) will result in an
    error.

    Args:
        patient_code (str): The identification code for the patient.
        report_template (str): The template string containing placeholders.

    Returns:
        str: The patient's non-sensitive information formatted as per the
        template.

    Raises:
        ValueError: If the patient is not found, if arguments are of incorrect
            type, or if the template attempts to access sensitive information.
    """
    # Validate argument types early for clearer error messages.
    if not isinstance(patient_code, str):
        raise ValueError("patient_code must be a string.")
    if not isinstance(report_template, str):
        raise ValueError("report_template must be a string.")

    # Retrieve the patient record by code.
    patient = fetch_patient_data(patient_code, patients)

    # Render the final report based on the provided template.
    return process_report_template(report_template, patient)
