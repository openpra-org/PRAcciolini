class XMLRegistry:
    _tag_to_class = {}
    _class_to_tags = {}

    @classmethod
    def register(cls, class_obj, tags):
        for tag in tags:
            cls._tag_to_class[tag] = class_obj
            cls._class_to_tags.setdefault(class_obj, []).append(tag)

    @classmethod
    def get_class_by_tag(cls, tag):
        return cls._tag_to_class.get(tag)

    @classmethod
    def get_tags_by_class(cls, class_obj):
        return cls._class_to_tags.get(class_obj, [])