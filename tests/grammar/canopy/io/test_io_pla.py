import os
import unittest
import flatbuffers

from pracciolini.grammar.canopy.io.pla import PLAType, PLAStart, PLAStartProductsVector, PLAAddNumProducts, \
    PLAAddNumEventsPerProduct, PLAAddType, PLAAddProducts, PLAEnd, PLA

class TestIOCanopyPLA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get the directory of the current script
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the base path for fixtures
        cls.fixtures_path = os.path.join(cls.current_dir, '../../../fixtures/canopy/pla/valid')


    def test_build_and_write_pla_to_file(self):
        builder = flatbuffers.Builder(0)

        # Products vector as defined in the set_F function
        # Encoding the product terms into bit vectors (uint8)
        # Values correspond to:
        # [0b01100111, 0b10011111, 0b11011011, 0b10011011, 0b00110011]
        products = [0b01100111, 0b10011111, 0b11011011, 0b10011011, 0b00110011]

        # Start the products vector in the builder
        PLAStartProductsVector(builder, len(products))
        # Prepend the products in reverse order, as required by FlatBuffers
        for value in reversed(products):
            builder.PrependUint8(value)
        products_vector = builder.EndVector()

        # Start building the PLA table
        PLAStart(builder)
        # Set the type to DNF (Disjunctive Normal Form)
        PLAAddType(builder, PLAType.DNF)
        # Add the products vector to the PLA
        PLAAddProducts(builder, products_vector)
        # Set the number of products
        PLAAddNumProducts(builder, len(products))
        # Set the number of events per product (number of bits used per product term)
        PLAAddNumEventsPerProduct(builder, 8)
        # End the PLA table
        pla = PLAEnd(builder)

        # Finish the buffer with the PLA as the root object and set the file identifier
        builder.Finish(pla, file_identifier=b'BPLA')

        # Get the output buffer ready to write to a file
        buf = builder.Output()

        # Write the buffer to a file named 'pla_output.bits'
        file_path = '/tmp/pla_output.bits'
        with open(file_path, 'wb') as f:
            f.write(buf)

        print('PLA FlatBuffer successfully built and written to ', file_path)

        self.test_read_and_validate_pla_from_file(file_path)

    def test_read_and_validate_pla_from_file(self, file_path: str = None):
        # Read the flatbuffer from 'pla_output.bits'
        if file_path is None:
            file_path = os.path.join(self.fixtures_path, 'dnf_8symbols_5terms.bits')
        self.assertTrue(os.path.exists(file_path), f"File '{file_path}' does not exist. Run the write test first.")

        with open(file_path, 'rb') as f:
            buf = f.read()

        # Verify that the buffer has the correct file identifier
        if not PLA.PLABufferHasIdentifier(buf, 0):
            self.fail("The buffer does not have the correct file identifier 'BPLA'.")

        # Get the root as PLA object
        pla = PLA.GetRootAs(buf, 0)

        # Verify the Type
        self.assertEqual(pla.Type(), PLAType.DNF, "PLA Type does not match PLAType.DNF")

        # Verify the NumProducts
        self.assertEqual(pla.NumProducts(), 5, "Number of products does not match expected value 5.")

        # Verify the NumEventsPerProduct
        self.assertEqual(pla.NumEventsPerProduct(), 8, "Number of events per product does not match expected value 8.")

        # Verify the Products vector
        expected_products = [0b01100111, 0b10011111, 0b11011011, 0b10011011, 0b00110011]
        self.assertEqual(pla.ProductsLength(), len(expected_products), f"Products length {pla.ProductsLength()} does not match expected {len(expected_products)}.")

        for i in range(pla.ProductsLength()):
            product = pla.Products(i)
            self.assertEqual(product, expected_products[i], f"Product at index {i} does not match expected value.")

        print('PLA FlatBuffer successfully read and validated from', file_path)

if __name__ == '__main__':
    unittest.main()
