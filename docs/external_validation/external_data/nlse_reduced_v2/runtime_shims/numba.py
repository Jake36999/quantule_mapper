def njit(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    def deco(fn):
        return fn
    return deco

def prange(*args):
    return range(*args)
