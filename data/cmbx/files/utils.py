from collections import UserDict
import time
from os import path
from datetime import timedelta


class Timer:
    """Print a description of what's happening and the time that it took."""

    def __init__(self, description=None):
        self.description = description

    def __enter__(self):
        self.t1 = time.perf_counter()
        if self.description is not None:
            print(self.description)

    def __exit__(self, *args):
        self.t2 = time.perf_counter()
        dt = timedelta(seconds=self.t2 - self.t1)
        print(f"{self.description}  Done in {dt}")


def preprocess_yaml(filename):
    """Simple YAML preprocessor to include extra files."""
    base_dir = path.dirname(filename)
    with open(filename) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith("#!include "):
            include_file = line.removeprefix("#!include").strip()
            if path.isfile(include_file):
                pass
            elif path.isfile(path.join(base_dir, include_file)):
                include_file = path.join(base_dir, include_file)
            else:
                raise RuntimeError(f"could not find file to include: {include_file}")
            print("including", include_file)
            with open(include_file) as f:
                include_lines = f.readlines()
            new_lines.extend(include_lines)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


def get_ul_key(dict_like, key):
    """Get a value from a dict-like object using a case-insensitive key."""
    key_list = list(dict_like.keys())
    key_list_lower = [k.lower() for k in key_list]
    if key.lower() not in key_list_lower:
        raise KeyError(f"could not find {key} in {dict_like}")
    ind = key_list_lower.index(key.lower())
    return dict_like[key_list[ind]]


def parse_tracer_bin(tracer_bin_key):
    """Takes a string of the form tracer_name_{int} and returns tracer_name, int."""
    key_split = tracer_bin_key.split('_')
    tracer_name = '_'.join(key_split[:-1])
    tracer_bin = int(key_split[-1])
    return tracer_name, tracer_bin


def parse_cl_key(cl_key):
    tracer_bin_keys = cl_key.split(', ')
    return list(map(parse_tracer_bin, tracer_bin_keys))


class PairCache(UserDict):
    """Simple class to keep track of all of the cross-spectra associated
        with a set of tracers and avoid recomputing them."""
    def __init__(self, obj_dict, operation):
        self.obj_dict = obj_dict
        self.func = operation
        self.data = {}

    def _standardize_key(self, key):
        return ", ".join(sorted(key.split(", ")))

    def _reverse_key(self, key):
        return ", ".join(reversed(key.split(", ")))

    def __getitem__(self, key):
        reversed_key = self._reverse_key(key)
        if key in self.data.keys():
            return self.data[key]
        if reversed_key in self.data.keys():
            return self.data[reversed_key]
        # compute and save if combination not found
        obj1, obj2 = key.split(", ")
        #val = self.func(self.obj_dict[obj1], self.obj_dict[obj2])
        val = self.func(self.obj_dict, obj1, obj2)
        self.__setitem__(key, val)
        return val

    def __setitem__(self, key, val):
        standardized_key = self._standardize_key(key)
        self.data[standardized_key] = val
