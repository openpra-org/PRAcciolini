class FileFilterFTREX(object):

    @staticmethod
    def model_types(self) -> dict[str, list[str]]:
        return {
            'fault_tree': ['ftl', 'ftp']
        }
