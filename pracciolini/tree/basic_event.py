class BasicEvent(object):
    """Representation of a basic event in a fault tree.

    Attributes:
        prob: Probability of failure of this basic event.
    """

    def __init__(self, name, prob):
        """Initializes a basic event node.

        Args:
            name: Identifier of the node.
            prob: Probability of the basic event.
        """
        super(BasicEvent, self).__init__(name)
        self.prob = prob

    def to_xml(self):
        """Produces the Open-PSA MEF XML definition of the basic event."""
        return ("<define-basic-event name=\"" + self.name + "\">\n"
                "<float value=\"" + str(self.prob) + "\"/>\n"
                "</define-basic-event>\n")

    def to_aralia(self):
        """Produces the Aralia definition of the basic event."""
        return "p(" + self.name + ") = " + str(self.prob) + "\n"