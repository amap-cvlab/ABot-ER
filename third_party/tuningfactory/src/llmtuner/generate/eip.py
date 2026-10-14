from .utils import singleton


@singleton
class EIPSingleInstance(object):
    def __init__(self):
        self.instance = None

    def set_EIP_instance(self, instance):
        self.instance = instance

    def trace(self, data, slice_info=None):
        if slice_info:
            self.instance.trace_with_slice_info(slice_info, data)
        else:
            self.instance.trace(data)

    def close(self):
        self.instance.__exit__(None, None, None)
