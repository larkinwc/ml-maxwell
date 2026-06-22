"""vllm-maxwell: out-of-tree vLLM platform plugin for NVIDIA Maxwell (sm_50).

Entry point ``vllm.platform_plugins`` -> ``register``. vLLM calls ``register()``
in every worker process; it returns the fully-qualified Platform class name when
the current machine actually has a Maxwell GPU, else ``None`` (so the plugin is a
no-op on non-Maxwell hosts and never interferes with day-0 upstream behavior).
"""


def _is_maxwell() -> bool:
    """True iff at least one visible CUDA device is Maxwell (cc 5.x)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        for i in range(torch.cuda.device_count()):
            major, _ = torch.cuda.get_device_capability(i)
            if major == 5:
                return True
        return False
    except Exception:
        return False


def register():
    """vLLM platform-plugin entry point.

    Return the Platform class FQN only on real Maxwell hardware. Returning None
    elsewhere keeps the plugin inert so upstream vLLM is unaffected.
    """
    if not _is_maxwell():
        return None
    return "vllm_maxwell.platform.MaxwellPlatform"
