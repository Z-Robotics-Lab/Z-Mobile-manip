# NUC thin-deploy contract

The two hosts do not run the same Python package. The 4090 imports the full `z_manip`
(Pinocchio IK, CasADi QP, planning); the onboard NUC runs a **thin package** carrying only
what a short-lived executor needs, and deliberately omits Pinocchio and CasADi. The decision
loop lives on the 4090 — the NUC only executes.

`scripts/runtime/install_go2w_reactive_runtime.sh` SCPs the reactive executors plus a reduced
`z_manip/` tree into `~/.local/lib/z-mobile-manip/`. The contract is what it does next: it
copies `configs/nuc-kinematics-init.py` and `configs/nuc-control-init.py` and **renames them to
`__init__.py`** inside `z_manip/kinematics/` and `z_manip/control/`. Those stubs export only the
light surface — `KinematicChain` from `chain.py`, the posture transport marker — so importing
`z_manip.kinematics` or `z_manip.control` on the NUC never pulls the heavy dependency chain,
while the executors still resolve the modules they need.

- **Do not** deploy the full package `__init__.py` to the NUC. It drags in Pinocchio/CasADi and
  breaks the thin-host assumption.
- Edit the stubs at `configs/nuc-{kinematics,control}-init.py`, not on the NUC — the deploy
  renames them in place, so an edit made there is overwritten by the next deploy.
- Passive CAN telemetry (read-only `/piper/state` at 20 Hz) is a separate, non-owning deploy:
  `scripts/runtime/install_nuc_passive_access.sh`.

Cross-machine rationale lives in go2w-integration
[`docs/reproduction.md` §A.5](https://github.com/Z-Robotics-Lab/go2w-integration/blob/main/docs/reproduction.md)
and [`docs/modules.md` module C](https://github.com/Z-Robotics-Lab/go2w-integration/blob/main/docs/modules.md).
