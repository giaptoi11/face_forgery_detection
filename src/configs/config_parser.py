import yaml
from types import SimpleNamespace
import re

def load_config(path):
    """
    Load YAML config and return as a nested SimpleNamespace object.
    Auto-converts strings like '1e-5' to float.
    """
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    sci_float_pattern = re.compile(r"^[+-]?\d+(\.\d+)?[eE][+-]?\d+$")

    def convert_value(v):
        if isinstance(v, str) and sci_float_pattern.match(v):
            try:
                return float(v)
            except ValueError:
                return v
        elif isinstance(v, dict):
            return {k: convert_value(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [convert_value(x) for x in v]
        else:
            return v

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        else:
            return d

    cfg_converted = convert_value(cfg_dict)
    return dict_to_namespace(cfg_converted)
