import json
from pathlib import Path
import platform
import sys
import sysconfig

result = {
    "machine": platform.machine(),
    "python": sys.version,
    "executable": sys.executable,
    "config": {key: sysconfig.get_config_var(key) for key in
               ("VERSION", "SOABI", "EXT_SUFFIX", "MULTIARCH", "Py_ENABLE_SHARED", "SIZEOF_VOID_P", "Py_DEBUG")},
}
assert result["machine"] == "aarch64"
assert sys.version_info[:2] == (3, 11)
assert result["config"]["EXT_SUFFIX"] == ".cpython-311-aarch64-linux-gnu.so"
Path(__file__).with_name("python-probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
