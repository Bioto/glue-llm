"""Hook system for GlueLLM.

This module provides hook functionality for intercepting and transforming
data before and after LLM processing.
"""

from gluellm.hooks.manager import (
    GLOBAL_HOOK_REGISTRY,
    HookManager,
    clear_global_hooks,
    register_global_hook,
    unregister_global_hook,
)

__all__ = [
    "HookManager",
    "GLOBAL_HOOK_REGISTRY",
    "register_global_hook",
    "unregister_global_hook",
    "clear_global_hooks",
]
