import os
import sys

# Ensure hooks are importable and enabled by default
os.environ.setdefault("MOE_HOOK_ENABLE", "1")
os.environ.setdefault("MOE_HOOK_CONFIG", "/home/ecnu/disk/wzq/moe_hook_config.yaml")

# Import and apply hooks before launching server
from moe_hook.hooks import install_hooks
install_hooks()

# Delegate to sglang launcher (mirror launch_server.__main__ behavior)
from sglang.launch_server import prepare_server_args, run_server  # type: ignore


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        from sglang.srt.utils import kill_process_tree  # type: ignore

        kill_process_tree(os.getpid(), include_parent=False)
