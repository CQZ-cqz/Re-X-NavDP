"""Print the carb settings relevant to the home-scene MDL render deadlock."""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True, device="cuda:0")
import carb

settings = carb.settings.get_settings()
paths = [
    "/app/asyncRendering",
    "/app/asyncRenderingLowLatency",
    "omni.replicator.asyncRendering",
    "app.renderer.waitIdle",
    "app.hydraEngine.waitIdle",
    "app.execution.debug.forceSerial",
    "app.updateOrder.checkForHydraRenderComplete",
    "/physics/updateToUsd",
]
for p in paths:
    try:
        print(f"{p} = {settings.get(p)!r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{p} = ERROR {e!r}", flush=True)

# Try setting async rendering off and verify
settings.set_bool("/app/asyncRendering", False)
settings.set_bool("/app/asyncRenderingLowLatency", False)
print("after set:", flush=True)
for p in ("/app/asyncRendering", "/app/asyncRenderingLowLatency"):
    print(f"  {p} = {settings.get(p)!r}", flush=True)

app_launcher.app.close()
