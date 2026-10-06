"""P26 MIA host wrapper with the P24 detector override runtime.

The tracker, first-frame XML initialization, and cross-view MIA logic are the
sealed P26 wrapper.  Only the import is redirected to the runtime that reads
DET_OVERRIDE_DIR/<sequence>/<frame>.npy; this keeps detector arrays frozen
before the host run and leaves the host association code unchanged.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
P26_WRAPPER = HERE.parent / "p26_mia_baseline" / "wrappers" / "supplement_mia.py"
source = P26_WRAPPER.read_text(encoding="utf-8")
source = source.replace(
    "from mdmt_compat_runtime import init_model_mdmt as init_model",
    "from p30_override_runtime import init_model_mdmt as init_model",
    1,
)
if "p30_override_runtime" not in source:
    raise RuntimeError("runtime import replacement failed")
exec(compile(source, str(P26_WRAPPER), "exec"), globals(), globals())
