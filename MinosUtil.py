import numpy as np

def ternary(cond, a, b):
    """
    TERNARY Ternary operator

    c = ternary(cond, a, b) returns a if cond is true and b otherwise.
    """
    # Check if the condition is empty and treat it as False
    if cond is None:
        cond = False
    
    # Check if the condition is a scalar
    if np.isscalar(cond) or isinstance(cond, bool):
        return a if cond else b
    else:
        raise ValueError('Condition has to be logical or a scalar')
    

def str2int(txt, format_spec="%d32"):
    """
    Convert string to integer correctly for values above flintmax equivalent.
    
    Parameters:
    - txt: Input text to convert. Can be a string or a list of strings.
    - format_spec: Format specifier string, e.g., "%d64".
    
    Returns:
    - x: Numeric array of integers or a single integer.
    """
    # Default format_spec if not provided
    if format_spec is None or format_spec == "":
        format_spec = "%d32"

    if txt is None:
        return None

    # If txt is not a list, convert it to a string
    if isinstance(txt, (list, tuple)):
        return [str2int(t, format_spec) for t in txt]
    
    if isinstance(txt, str):
        txt = [txt]  # Convert to list for consistent processing

    # Update format_spec if it starts with "int" or "uint"
    if format_spec.startswith("int"):
        format_spec = "%d" + format_spec[3:]
    elif format_spec.startswith("uint"):
        format_spec = "%u" + format_spec[4:]
    
    # Ensure format_spec starts with "%"
    if not format_spec.startswith("%"):
        format_spec = "%" + format_spec

    # Verify that format_spec represents an integer
    if format_spec[1] not in ["d", "u", "x", "b"]:
        raise ValueError(f"Format specification {format_spec} is not an integer")
    
    # Perform conversion
    try:
        x = [int(t) for t in txt]
        if len(x) == 1:
            return x[0]  # Return a scalar if input was scalar
        else:
            return x  # Return list of converted values
    except ValueError as e:
        raise ValueError(f"Error converting {txt} with format {format_spec}: {e}")
