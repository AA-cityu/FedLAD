_PARSER_REGISTRY = {}


def register_parser(name):
    def decorator(func):
        _PARSER_REGISTRY[name.lower()] = func
        return func
    return decorator


def get_parser(name):
    name = name.lower()
    if name not in _PARSER_REGISTRY:
        raise ValueError(f"Parser for dataset '{name}' is not registered.")
    return _PARSER_REGISTRY[name]
