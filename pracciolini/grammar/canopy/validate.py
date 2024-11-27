import os

from pracciolini.core.decorators import validation
from pracciolini.grammar.canopy.io.pla import PLA, PLAType

@validation("canopy_pla")
def validate_pla_file(file_path: str) -> bool:
    """
    Validates that the data in the provided FlatBuffer file matches the PLA class schema.

    Args:
        file_path (str): The path to the FlatBuffer file to validate.

    Returns:
        bool: True if the file is valid according to the PLA schema, False otherwise.
    """

    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return False

    try:
        # Read the FlatBuffer from the file
        with open(file_path, 'rb') as f:
            buf = f.read()

        # Verify that the buffer has the correct file identifier
        if not PLA.PLABufferHasIdentifier(buf, 0):
            print("The buffer does not have the correct file identifier 'BPLA'.")
            return False

        # Initialize the root as a PLA object
        pla = PLA.GetRootAs(buf, 0)

        # Validate the 'Type' field
        pla_type = pla.Type()
        if pla_type not in (PLAType.DNF, PLAType.CNF):
            print(f"Invalid PLA Type: {pla_type}. Expected PLAType.DNF ({PLAType.DNF}) or PLAType.CNF ({PLAType.CNF}).")
            return False

        # Validate the 'NumProducts' field
        num_products = pla.NumProducts()
        if num_products <= 0:
            print(f"Invalid NumProducts: {num_products}. It should be a positive integer.")
            return False

        # Validate the 'NumEventsPerProduct' field
        num_events_per_product = pla.NumEventsPerProduct()
        if num_events_per_product <= 0:
            print(f"Invalid NumEventsPerProduct: {num_events_per_product}. It should be a positive integer.")
            return False

        # Validate the 'Products' vector
        if pla.ProductsIsNone():
            print("Products vector is None.")
            return False

        products_length = pla.ProductsLength()
        if products_length != num_products:
            print(f"Products length ({products_length}) does not match NumProducts ({num_products}).")
            return False

        for i in range(products_length):
            product = pla.Products(i)
            if not isinstance(product, int):
                print(f"Product at index {i} is not an integer.")
                return False
            if not (0 <= product <= 255):
                print(f"Product at index {i} ({product}) is out of uint8 range (0-255).")
                return False

        # If all validations pass
        print("PLA FlatBuffer file is valid and matches the PLA class schema.")
        return True

    except Exception as e:
        print(f"Exception during validation: {e}")
        return False
