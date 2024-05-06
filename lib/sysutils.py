import sys, os, importlib.util

_module_t = type(sys)

def include_path(p, globals_=None) -> None:
    "execute code directly here. default for globals_ is `sys._getframe(1).f_globals`"
    fname = os.path.basename(p)
    with open(p) as f: src = f.read()
    code = compile(src, fname, 'exec')
    exec(code, sys._getframe(1).f_globals if globals_ is None else globals_)

def import_path(p)->_module_t:
    """
    `mod = import_path(module_path)` is equivalent to `import module as mod`
    ! except no caching is done !
    """
    name = os.path.basename(os.path.splitext(p)[0])
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def import_path_star(p, globals_ = None)->_module_t:
    """
    analog of star import expression `from module import * `.
    ! except no caching is done !
    globals_: dict that is updated with imported names,
       default is `sys._getframe(1).f_globals`
    """
    mod = import_path(p)
    globals_ = sys._getframe(1).f_globals if globals_ is None else globals_
    d = mod.__dict__
    if hasattr(mod, '__all__'):
        for k in mod.__all__: globals_[k] = d[k]
    else:
        for k, v in d.items():
            if k.startswith("_"): continue
            globals_[k] = v
    return mod


def get_frameinfo(framelvl=1):
    """
    return info about the caller line, eg:
    ```
    def my_fn():
        print(get_frameinfo())
    my_fn()
    ```
    would print something like 
    `{'file': '/tmp/ipykernel_93217/1851559004.py', 'fn': 'my_fn', 'line': 2}`
    """
    f = sys._getframe(framelvl)
    return dict(file=f.f_code.co_filename, fn=f.f_code.co_name, line=f.f_lineno)


def get_caller(go_up_the_stack=0):
    'a fun piece of meta, that gets reference to caller function'
    n = sys._getframe(go_up_the_stack+1).f_code.co_name
    f = sys._getframe(go_up_the_stack+2)
    return f.f_locals.get(n, False) or f.f_globals.get(n)


def mylog1(lvl, msg: str):
    'little logger see dir(mylog1) for config attributes'
    f = mylog1
    if type(lvl) == str: lvl = f.leveldict[lvl]
    msg = f"[{lvl}] {get_frameinfo(2)} {msg}\n"
    if lvl <= f.maxlevel:
        f.stream.write(msg)
    if len(f.log) >= f.max_entries:
        f.log.pop(0)
    if f.max_entries > 0:
        f.log.append(msg)
    return msg

mylog1.log = []
mylog1.stream = sys.stderr
mylog1.max_entries = 100
mylog1.maxlevel = 2
mylog1.leveldict = dict(debug=3, info=2, error=1)

log = mylog1