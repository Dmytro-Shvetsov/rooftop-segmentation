import os
from copy import deepcopy
from typing import Any
import pydoc
from functools import reduce

import yaml
import shutil


class Config:

    def __init__(self, file_path):
        self._orig_fp = file_path
        self._load(file_path)

    def _load(self, file_path):
        with open(file_path, 'r') as fid:
            cfg = yaml.load(fid, yaml.Loader)
        cfg_obj = cfg.get('config')
        if cfg_obj is not None:
            cfg = cfg_obj.as_dict()
        self.__dict__.update(cfg)

    def get(self, param_name, default=None, copy=False):
        ret = self.__dict__.get(param_name, default)
        return deepcopy(ret) if copy else ret

    def as_dict(self):
        return deepcopy(self.__dict__)

    def update(self, new_dict):
        for k, v in new_dict.items():
            self.__setitem__(k, v)

    def pop(self, key, default):
        return self.__dict__.pop(key, default)

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return reduce(lambda c, k: c.get(k, {}), key.split('.'), self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, other):
        return other in self.__dict__

    def clear(self):
        return self.__dict__.clear()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def copy(self, fp):
        if os.path.exists(fp):
            os.unlink(fp)
        shutil.copy(self._orig_fp, fp)

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            raise AttributeError(f'Config parameter {repr(__name)} not found. Available parameters: {list(self.__dict__.keys())}')


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop('type', None)
    if object_type is None:
        raise ModuleNotFoundError(f'Unable to find a module {repr(object_type)} with missing type. Module parameters: {list(d.keys())}')

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)
