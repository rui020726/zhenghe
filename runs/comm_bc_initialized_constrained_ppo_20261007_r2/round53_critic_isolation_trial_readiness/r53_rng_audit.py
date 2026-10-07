"""Read-only hashes of the registered ROUND44 ep0 RNG payloads; never restores state."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_registered_rng(path: Path, *, expected_file_sha256: str) -> dict[str, Any]:
    """Load an existing checkpoint on CPU and hash only its saved RNG components.

    This routine intentionally contains no random seeding, ``set_state`` or
    ``restore_rng`` call, no model forward, and no checkpoint write.
    """
    path = Path(path)
    file_sha = _sha(path.read_bytes())
    if file_sha != expected_file_sha256:
        raise RuntimeError(f"registered ep0 RNG checkpoint hash mismatch: {path}")
    import numpy as np
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    rng = payload.get("rng")
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch"}:
        raise RuntimeError(f"registered checkpoint RNG components mismatch: {path}")
    python_bytes = pickle.dumps(rng["python"], protocol=4)
    numpy_state = rng["numpy"]
    if (not isinstance(numpy_state, (tuple, list)) or len(numpy_state) != 5
            or numpy_state[0] != "MT19937"):
        raise RuntimeError(f"registered NumPy RNG state schema mismatch: {path}")
    numpy_array = np.asarray(numpy_state[1], dtype="<u4").tobytes(order="C")
    numpy_bytes = (str(numpy_state[0]).encode("ascii") + b"\0" + numpy_array
                   + pickle.dumps(tuple(numpy_state[2:]), protocol=4))
    torch_state = rng["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_bytes = torch_state.detach().cpu().contiguous().numpy().tobytes(order="C")
    else:
        torch_bytes = bytes(torch_state)
    return {
        "path": str(path.resolve()), "checkpoint_sha256": file_sha,
        "rng_component_keys": sorted(rng),
        "python_state_sha256_pickle_protocol4": _sha(python_bytes),
        "python_state_bytes": len(python_bytes),
        "numpy_state_sha256_mt19937_u32le_plus_tail_protocol4": _sha(numpy_bytes),
        "numpy_state_array_sha256_u32le": _sha(numpy_array),
        "numpy_state_array_elements": len(numpy_state[1]),
        "numpy_state_position_and_gauss_tail": list(numpy_state[2:]),
        "torch_state_sha256_raw_cpu_uint8_bytes": _sha(torch_bytes),
        "torch_state_bytes": len(torch_bytes),
        "restored_or_set": False,
    }
