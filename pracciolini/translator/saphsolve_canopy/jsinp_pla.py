import flatbuffers

from pracciolini.core.decorators import translation
from pracciolini.grammar.canopy.io.pla import PLA, PLAType, PLAStart, PLAStartProductsVector, PLAAddType, \
    PLAAddProducts, PLAAddNumProducts, PLAAddNumEventsPerProduct, PLAEnd
from pracciolini.grammar.saphsolve.jsinp.jsinp import JSInp

@translation('saphsolve_jsinp', 'canopy_pla')
def saphsolve_jsinp_to_canopy_pla(jsinp: JSInp, type: PLAType = PLAType.DNF) -> PLA:
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
    generated_pla: PLA = PLA()
    #generated_pla.Type() =
    return generated_pla