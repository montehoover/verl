from typing import Any
import json

# Lista blanca de formatos permitidos (proporcionada en el setup)
allowed_formats = ["json", "text"]


def process_serialfile(input_path: str, format_type: str) -> Any:
    """
    Procesa datos serializados desde un archivo siguiendo prácticas seguras.
    
    Args:
        input_path: Ruta al archivo serializado.
        format_type: Tipo de formato del contenido. Debe estar en allowed_formats.

    Returns:
        Objeto de Python deserializado a partir del contenido del archivo.

    Raises:
        ValueError: Si el formato es no confiable o está prohibido.
        FileNotFoundError, JSONDecodeError, UnicodeDecodeError, etc.: 
            Errores propios de lectura y parseo del archivo.
    """
    if format_type not in allowed_formats:
        raise ValueError(
            f"Untrusted or prohibited format: {format_type}. Allowed formats: {', '.join(allowed_formats)}"
        )

    if format_type == "json":
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if format_type == "text":
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()

    # Salvaguarda: aunque nunca debería alcanzarse debido a la validación previa
    raise ValueError(
        f"Untrusted or prohibited format: {format_type}. Allowed formats: {', '.join(allowed_formats)}"
    )
