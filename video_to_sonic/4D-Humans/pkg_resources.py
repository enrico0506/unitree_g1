from importlib.util import find_spec
from pathlib import Path


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    spec = find_spec(package_or_requirement)
    if spec is None:
        raise ModuleNotFoundError(f"Cannot resolve package {package_or_requirement!r}")

    if spec.submodule_search_locations:
        base = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin is not None:
        base = Path(spec.origin).parent
    else:
        raise FileNotFoundError(f"Cannot resolve resources for {package_or_requirement!r}")

    return str(base / resource_name)
