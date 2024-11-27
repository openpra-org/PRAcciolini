# import xml.etree.ElementTree as ET
# from collections import deque
# from xml.sax.saxutils import escape
#
# class EventTree:
#     def __init__(self, name):
#         self.name = name
#         self.functional_events_id = []
#         self.functional_events_name =[]
#         self.sequences = []
#         self.initial_state = {}
#
#     def to_xml(self):
#
#         event_tree_element = ET.Element('define-event-tree', {'name': self.name})
#
#         # Iterate over both functional events and their corresponding labels
#         for functional_event, functional_event_name in zip(self.functional_events_id, self.functional_events_name):
#             # Create the functional event element with the corresponding label
#             functional_event_element = ET.SubElement(event_tree_element, 'define-functional-event',
#                                                      {'name': functional_event})
#             label_element = ET.SubElement(functional_event_element, 'label')
#             label_element.text = escape(functional_event_name)  # Set label text from the corresponding name list
#
#         for sequence in self.sequences:
#             sequence_element = ET.SubElement(event_tree_element, 'define-sequence', {'name': sequence})
#
#         if self.initial_state:
#             initial_state_element = ET.SubElement(event_tree_element, 'initial-state')
#             self._build_initial_state_xml(self.initial_state, initial_state_element)
#
#         return event_tree_element
#     def _build_initial_state_xml(self, state, parent_element):
#         queue = deque([(state, parent_element)])  # Initialize a queue with the root element
#         processed_elements = set()  # Track processed elements to avoid duplication
#
#         while queue:
#             current_state, current_parent = queue.popleft()  # Dequeue the current element
#
#             # Check if the current_state has already been processed
#             if id(current_state) in processed_elements:
#                 continue  # Skip processing if the element has already been processed
#             processed_elements.add(id(current_state))
#
#             # Ensure the state dictionary has the 'name' key
#             state_name = current_state.get('name', 'unknown_element')
#             # Create the current element with its attributes
#             current_element = ET.SubElement(current_parent, state_name, current_state.get('attributes', {}))
#
#             # Iterate over the children of the current state
#             for child in current_state.get('children', []):
#                 # Enqueue child elements for processing
#                 queue.append((child, current_element))
#
#
# class FaultTreeOpenPSA:
#     def __init__(self, name):
#         self.name = name
#         self.elements = []
#
#     def add_gate(self, name, gate_type):
#         # Ensure gate_inputs is always stored as a list
#         # if not isinstance(gate_inputs, list):
#         #     gate_inputs = [gate_inputs]
#         self.elements.append({'type': 'gate', 'name': name, 'gate_type': gate_type})
#         # self.elements.append(
#         #     {'type': 'gate', 'name': name, 'gate_type': gate_type, 'gate_inputs': gate_inputs, 'sub_elements': []})
#
#     def add_gate_input(self, name, gate_inputs):
#         if not isinstance(gate_inputs, list):
#             gate_inputs = [gate_inputs]
#         self.elements.append({'type': 'gate_input', 'name': name, 'gate_inputs': gate_inputs})
#
#     def add_basic_event(self, name, event_inputs):
#         # Ensure gate_inputs is always stored as a list
#         if not isinstance(event_inputs, list):
#             event_inputs= [event_inputs]
#         self.elements.append({'type': 'basic-event', 'name': name, 'event_inputs': event_inputs})
#
#     def add_comp_event(self, name, complement_event_inputs):
#         # Ensure gate_inputs is always stored as a list
#         if not isinstance(complement_event_inputs, list):
#             complement_event_inputs= [complement_event_inputs]
#         self.elements.append({'type': 'comp-event', 'name': name, 'event_inputs': complement_event_inputs})
#
#     def add_relationship(self, parent, child):
#         for element in self.elements:
#             if element['name'] == parent:
#                 element['sub_elements'].append({'type': 'relation', 'name': child})
#                 break
#
#     def to_xml(self):
#         global gate_type_element
#         fault_tree_element = ET.Element('define-fault-tree', {'name': self.name})
#
#         for element in self.elements:
#             element_type = element['type']
#             element_name = element['name']
#             # element_element = ET.SubElement(fault_tree_element, element_type, {'name': element_name})
#
#             if element_type == 'gate':
#                 gate_type = element['gate_type']
#                 gate_element = ET.SubElement(fault_tree_element, 'define-gate', {'name': element_name, 'role': 'private'})
#
#                 if gate_type == 'or' or gate_type == 'and':
#                     # Handle OR and AND gates as before
#                     gate_type_element = ET.SubElement(gate_element, gate_type)
#                 else:
#                     # Handle custom gate types like "2/3"
#                     gate_type_element = ET.SubElement(gate_element, 'atleast', {'min': gate_type.split('/')[0]})
#                     if element_type == "gate_input":
#                         for input_name in element['gate_inputs']:
#                             if input_name is not None:
#                                 ET.SubElement(gate_type_element, 'gate', {'name': input_name})
#                     elif element_type == "basic_event":
#                         for input_name in element['event_inputs']:
#                             if input_name is not None:
#                                 ET.SubElement(gate_type_element, 'basic-event', {'name': input_name})
#
#
#             elif element_type == 'gate_input':
#                 for input_name in element['gate_inputs']:
#                     if input_name is not None:
#                         ET.SubElement(gate_type_element, 'gate', {'name': input_name})
#
#             elif element_type == 'basic-event':
#                 for input_name in element['event_inputs']:
#                     if input_name is not None:
#                         ET.SubElement(gate_type_element, 'basic-event', {'name': input_name})
#
#             elif element_type == 'comp-event':
#                 for input_name in element['event_inputs']:
#                     if input_name is not None:
#                         comp_event_element = ET.SubElement(gate_type_element, 'not')
#                         ET.SubElement(comp_event_element, 'basic-event', {'name': input_name})
#         return fault_tree_element
#
#
#
# class ModelData:
#     def __init__(self):
#         self.basic_events = []
#
#     def add_basic_event(self, name, label=None, float_value=None):
#         self.basic_events.append({'name': name, 'label': label, 'float_value': float_value})
#
#     def to_xml(self):
#         model_data_element = ET.Element('model-data')
#
#         for basic_event_data in self.basic_events:
#             basic_event_element = ET.SubElement(model_data_element, 'define-basic-event', {'name': basic_event_data['name']})
#
#             if basic_event_data.get('label'):
#                 label_element = ET.SubElement(basic_event_element, 'label')
#                 label_element.text = escape(basic_event_data['label'])
#
#             if basic_event_data.get('float_value'):
#                 # Format the float value to ensure proper display
#                 # float_value = "{:.6E}".format(basic_event_data['float_value'])
#                 float_value = (basic_event_data['float_value'])
#                 float_element = ET.SubElement(basic_event_element, 'float', {'value': float_value})
#
#         return model_data_element
#
