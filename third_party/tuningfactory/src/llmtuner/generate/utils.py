import functools
import os

def singleton(cls):
    """
    This decorator will change the decorated class into singleton.
    Each time of the initialization of the decorated class will return the same object as the first time.

    Caution: This decorator will change the decorated class into function.
    Such that if you calling the decorated class's `super` method by `super(__class__, <first argument>)` would fail.
    But `super()` works just fine.

    Example:
    ```
    class TestBase:
        def __init__(self):
            print('TestBase')


    @singleton
    class Test(TestBase):
        def __init__(self):
            print(Test)
            super(Test, self).__init__()
            print('Test')
    ```
    You will see that `Test` becomes a function instead of class: `<function Test at 0x7fd59b994ee0>`
    And this code will generate an error: `TypeError: super() argument 1 must be type, not function`

    :param cls: The class to be decorated
    :return: function!!!
    """

    _instance = {}

    @functools.wraps(cls)
    def materialize(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = {'instance': cls(*args, **kwargs), 'counter': 0}
            # print(f'Initialized {cls} for the first time.')
        else:
            _instance[cls]['counter'] += 1
            # print(f'Returned instance of {cls} {_instance[cls]["counter"]} time(s).')
        return _instance[cls]['instance']
    return materialize

def is_external_cluster():
    is_external = os.environ.get('IS_EXTERNAL_CLUSTER', None)
    return str(is_external).lower() == 'true'