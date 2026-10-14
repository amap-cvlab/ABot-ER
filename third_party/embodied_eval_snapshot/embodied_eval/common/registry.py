
MODEL_REGISTRY = {}

def register_model(*names):

    def decorate(cls):
        for name in names:
            MODEL_REGISTRY[name] = cls
        return cls

    return decorate