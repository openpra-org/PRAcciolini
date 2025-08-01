import os
import json
from typing import Any, Tuple, List, Optional

from jsonschema import validate, ValidationError

from pracciolini.core.decorators import validation, load, save
from pracciolini.utils.file_ops import FileOps


def validate_json(json_data: str, schema: Any) -> Tuple[bool, Optional[List[str]]]:
    try:
        validate(instance=json_data, schema=schema)
        return True, None
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON format: {e.msg}"]
    except ValidationError as e:
        return False, [e.message]

@load("saphsolve_jsinp_json", ".jsinp")
def read_jsinp_json(json_file_path: str) -> json:
    return FileOps.parse_json_file(json_file_path)

@validation("saphsolve_jsinp_json")
def validate_jsinp_json(data: json, jsinp_schema_path: str = 'schema/jsinp.schema.json') -> bool :
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, jsinp_schema_path)
        schema = FileOps.parse_json_file(schema_path)
        is_valid, messages = validate_json(data, schema)
    except ValidationError as error:
        print(error)
        return False
    return is_valid

@save("saphsolve_jsinp_json", ".jsinp")
def write_jsinp_json(data: json, json_file_path: str) -> None:
    FileOps.write_json_file(data, json_file_path)


# @translation('saphsolve_jsinp_json', 'saphsolve_jsinp')
# def as_jsinp(data: json) -> JSInp:
#     return JS
# def read_and_validate_jsinp( ) -> JSInp:
#     try:
#         json_data = FileOps.parse_json_file(json_file_path)
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         valid,_ = validate_saphsolve_jsinp(json_data, jsinp_schema_path)
#     except ValidationError as e:
#         return False, [e.message]

# read the file -> object
# validate the constructed objected -> validated object
# translate(validated_object_type_1) -> object_type_2
# validate the translated object -> validated_object_type_2
# write to file -> path for written file
