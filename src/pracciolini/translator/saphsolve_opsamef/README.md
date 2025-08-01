## SAPHSOLVE Model Conversion to OpenPSA MEF
```SCRAM```accepts basic event and gate names as strings,
it does not accept only numbers even if the number are in string format like ```"1"```.
So, some initials are added while converting the SAPHSOLVE model to SCRAM (OpenPSA MEF) model.

- ```BE``` for Basic Event
- ```G``` for Gate
- ```S``` for Sequence
- ```INIT``` for Initiating Event
- ```FE``` for Functional Event
- ```FT``` for Fault Tree

Some concerns and notes for future model conversion:
- ```FE``` should be in order in ```XML``` file.
- ```SAPHSOLVE``` model may include a ```gate``` has only one ```child```, which is not allowed in ```SCRAM```.
Remember to add the following `basic-event` definition to `model-data` to handle this problem:
```xml
<define-basic-event name="BE0">
    <label>TEMP-BE-TO-ADD-GATES-WITH-A-SINGLE-ELEMENT</label>
    <float value="0.000000E+00"/>
</define-basic-event>
```
- ```SAPHSOLVE``` model does not include ``FE`` order, which is required while building ```SCRAM``` ```initial-state```.
