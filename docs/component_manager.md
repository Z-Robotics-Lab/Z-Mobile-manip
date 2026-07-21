# Z-Manip vision component manager

`scripts/runtime/go2w_component_manager.sh` is the fixed process owner for the
loopback workbench and its camera/perception dependencies. It does not accept
commands, paths, ROS topics, or target poses from the browser.

## After a NUC or PC reboot

```bash
cd ~/Z-Robotics-Lab/Z-Mobile-manip
scripts/runtime/go2w_component_manager.sh bringup
scripts/runtime/go2w_component_manager.sh status all
```

`bringup` is idempotent and protected by one `flock`. It starts the NUC D435
service, establishes the authorized receive-only `can0` gate when the interface
is down, starts passive joint feedback, and then starts EdgeTAM, the RGB-D
bridge, perception ROS, the subscribe-only runtime observer, and the loopback
UI. Every stage has a bounded health wait. The passive bus gate verifies that
the host TX packet counter stays at zero.

Camera health is input-aware: `nuc-camera` is healthy only when the RealSense
USB device is present and `d435i.service` is active; `rgbd` is healthy only
when a fresh camera artifact is arriving. If the D435 is absent, reconnect it
before `bringup` or `restart nuc-camera`; the manager fails immediately with a
specific cable/device message.

The same action is available in the dashboard under **Vision services** as
**Bring up all vision services**.

## Restart one component

```bash
scripts/runtime/go2w_component_manager.sh restart edgetam
scripts/runtime/go2w_component_manager.sh restart rgbd
scripts/runtime/go2w_component_manager.sh restart perception
scripts/runtime/go2w_component_manager.sh restart observer
scripts/runtime/go2w_component_manager.sh restart perception-all
```

The dashboard exposes the same fixed allowlist. A restart is refused while
Home, perception, planning, or grasp is active. Restarting a vision component
clears the selected perception/plan context so stale overlays cannot be reused.

## Status and logs

```bash
scripts/runtime/go2w_component_manager.sh status all
scripts/runtime/go2w_component_manager.sh logs edgetam 100
scripts/runtime/go2w_component_manager.sh logs perception-all 100
scripts/runtime/go2w_component_manager.sh logs manager 100
```

Component logs are bounded to 300 requested lines and 24 KB of output. The UI
also suppresses routine successful polling access logs while retaining POSTs,
unexpected routes, and HTTP errors.
