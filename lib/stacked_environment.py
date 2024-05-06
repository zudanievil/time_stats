"""
provides two primitives (`dct_with`, `dct_scope`) to 
modify config dict in a safe (sort of) manner.
"""

__all__ = ["dct_with", "dct_scope", "dct_diff_update", "dct_getitems", "NSdict"]
_none = object()

def _dct_put(self: "{'dct', 'options'}") -> None:
    """
    `for k in self.options`, swap corresponding values of `self.dct`
    and `self.options`. unique object `_none` is put into `self.options`
    if `self.dct[k]` does not exist.
    """
    none = _none
    dct = self.dct
    opts = self.options
    for k, v1 in opts.items():
        v0 = dct.get(k, none)
        opts[k] = v0
        dct[k] = v1
def _dct_pop(self: "{'dct', 'options'}", *_) -> None:
    """reverse the action of `_dct_put`"""
    none = _none
    dct = self.dct
    opts = self.options
    for k, v0 in self.options.items():
        opts[k] = dct.pop(k)
        if v0 is not none:
            dct[k] = v0

def dct_getitems(dct: dict, keys: list):
    "get items in bulk (useful for validation)"
    r = [None]*len(keys)
    for i, k in enumerate(keys):
        r[i] = dct[k]
    return r

class dct_with:
    """
    so called 'context manager' that swaps values in `self.dct`
    for corresponding values in `self.options`.
    """
    __slots__ = ("dct", "options")
    def __init__(self, dct: dict, options: dict):
        self.dct = dct; self.options = options
    __enter__ = _dct_put
    put = _dct_put
    __exit__ = _dct_pop
    pop = _dct_pop

class dct_scope:
    """
    swaps values in `self.dct` for corresponding values in `self.options`
    until `self.__del__` is called (normally at the end of the scope)
    """
    __slots__ = ("dct", "options")
    def __init__(self, dct: dict, options: dict):
        self.dct = dct; self.options = options
        _dct_put(self)
    __del__ = _dct_pop

def _test_dct_with():
    e1 = dict(int=5, float=5.5, bool=True)
    ENV = e1.copy()
    e2 = dict(int=6, float=6.5, chr='a')
    ctx = dct_with(ENV, e2.copy())
    with ctx:
        assert ENV == dict(int=6, float=6.5, chr='a', bool=True), repr(ENV)
        assert ctx.options == dict(int=5, float=5.5, chr=_none), repr(ctx.options)
        # ^^^ implementation detail
    assert ENV == e1, f"{ENV} != {e1}"
    assert ctx.options == e2, f"{ctx.options}, {e2}"

def _test_dct_scope():
    e1 = dict(int=5, float=5.5, bool=True)
    ENV = e1.copy()
    e2 = dict(int=6, float=6.5, chr='a')
    e2_ctx = e2.copy() # not to reference ctx
    ctx = dct_scope(ENV, e2_ctx)
    assert ENV == dict(int=6, float=6.5, chr='a', bool=True), repr(ENV)
    assert e2_ctx == dict(int=5, float=5.5, chr=_none), repr(e2_ctx)
    # ^^^ implementation detail
    del ctx
    
    assert ENV == e1, f"{ENV} != {e1}"
    assert e2_ctx == e2, f"{e2_ctx}, {e2}"

def dct_diff_update(dst, src):
    for k, v in src.items():
        if k not in dst: dst[k] = v
    return dst


class NSdict(dict):
    """Namespace dict. like dict, but able to use obj.key = value & obj.key syntax"""
    def __getattr__(self, attr):
        try:
            v = self.__getattribute__(attr)
            return v
        except AttributeError: pass
        if attr.startswith("_"): raise NameError("key names cannot start with underscores `_`")
        return dict.__getitem__(self, attr)
    def __setattr__(self, attr, value):
        aexists = True # prevents chaining exceptions, which can be weird
        try: v = self.__getattribute__(attr)
        except AttributeError: aexists = False
        if aexists: raise AttributeError(f"{attr} is bound to {v!r}")
        else:
            if attr.startswith("_"): raise NameError("key names cannot start with underscores `_`")
            return dict.__setitem__(self, attr, value)

# ========= old version ========
# class StackedEnvironment:
#     """
#     inverse principle from how environments normally work: 
#         dict of lists instead of list of dicts.
#     this inverts how execution environments work: 
#         caller code can set environment of the callee,
#         creation scope is ignored.
#     it is intended to be used as a dynamic global configuration,
#     instead of ordinary dict. unlike dict it is not suitable 
#     for asynchronous mutation, and should be used in REPL or
#     in synchronous parts of the code.

#     supports most important parts of dict api:
#     	`__getitem__`, `__setitem__`, `get`, `items`, `keys`, `pop`
#     some additional utility methods:
#     	`__delitem__ = pop`, `put = __setitem__`, 
#     	`get_many`, `put_many`, `pop_many` 
#     has special methods:
#     	`with_set` -- apply changes at the start of `with`
#     	block, reset them at the end.
#     	`scope_set` -- apply changes, revert them when returned
#     	object destructor (`__del__`) is called (much like RAII),
#     	which normally is on function exit.
#     """
#     __slots__ = ("options", )
#     def __init__(self, options: dict[str]):
#         self.options = {k: [v, ] for k, v in options.items()}
#     def __getitem__(self, key: str):
#         return self.options[key][-1]
#     def get_many(self, keys: tuple[str, ...]):
#         return [self.get(k) for k in keys]
#     def get(self, key: str):
#         v = self.options.get(key)
#         return None if v is None else v[-1]
#     def pop(self, key: str):
#         v = self.options[key]
#         if len(v) > 1:
#             return v.pop(-1)
#         else:
#             return self.options.pop(key)[0]
#     __delitem__ = pop 
#     def pop_many(self, keys: tuple[str, ...]):
#         missing = [k for k in keys if not k in self.options]
#         if missing: raise KeyError(f"missing keys: {missing}")
#         for k in keys:
#             v = self.options.get(k)
#             if len(v) > 1:
#                 v.pop(-1)
#             else:
#                 self.options.pop(k)[0]
#     def put(self, key: str, value):
#         v = self.options.setdefault(key, [])
#         v.append(value)
#     __setitem__ = put
#     def put_many(self, options: dict[str]):
#         for k, v in options.items():
#             vs = self.options.setdefault(k, [])
#             vs.append(v)
#     def keys(self):
#         return self.options.keys()
#     def items(self):
#         return ((k, v[-1]) for k, v in self.options.items())
#     def top(self) -> dict[str]:
#         return {k: v[-1] for k, v in self.options.items()}
#     def items(self):
#         return self.options.items()
#     def __repr__(self):
#         return f"StackedEnvironment. top:\n{self.top()}"
#     class _with:
#         __slots__ = ("env", "options")
#         def __init__(self, env, options): 
#             self.env = env; self.options = options
#         def __enter__(self):
#             self.env.put_many(self.options)
#         def __exit__(self, _a, _b, _c):
#             self.env.pop_many(self.options)
#     class _scope:
#         __slots__ = ("env", "options")
#         def __init__(self, env, options):
#             self.env = env; self.options = options
#             self.env.put_many(self.options)
#         def __del__(self):
#             self.env.pop_many(self.options)
#     def with_set(self, options: dict[str]) -> _with:
#         return self._with(self, options)
#     def scope_set(self, options: dict[str]) -> _scope:
#         return self._scope(self, options)  


# def _test_with_set():
#     _env = StackedEnvironment(dict(int=5, float=7.0))
#     with _env.with_set(dict(int=9, float=11.3, alpha="a")):
#         assert _env.top() == {'int': 9, 'float': 11.3, 'alpha': 'a'}, _env.top()
#         _ = _env.get_many(("int", "float", "alpha"))
#         assert _ == [9, 11.3, 'a'], _
#     assert _env.top() == {'int': 5, 'float': 7.0}, _env.top()
#     _ = _env.get_many(("int", "float", "alpha", "beta"))
#     assert _ == [5, 7.0, None, None], _
# # _test_with_set()

# def _test_scope_set():
#     _env = StackedEnvironment(dict(int=5, float=7.5))
#     assert _env.top() == {'int': 5, 'float': 7.5}, _env.top()
#     def f():
#         __defer = _env.scope_set(dict(int=0, alpha='a'))       
#         assert _env.top() == {'int': 0, 'float': 7.5, 'alpha': 'a'}, _env.top()
#     f()
#     assert _env.top() == {'int': 5, 'float': 7.5}, _env.top()
# # _test_scope_set()

# ==============================

