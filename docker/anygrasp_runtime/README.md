# AnyGrasp runtime adapter

This is a secret-free image layered on the locally built `anygrasp:cu128-py311`
runtime. License files and the checkpoint remain bind-mounted from `~/anygrasp`.
The server exposes the same `z-manip.grasp.v1` msgpack/ZMQ contract consumed by
`GraspInferenceClient` and converts GraspNet axes to the stack's TCP convention.

Use `scripts/runtime/go2w_perception_lab.sh anygrasp-build`, replace the
machine-bound license, then run `anygrasp-start`. The start command refuses to
launch the server unless the SDK license/CUDA verification succeeds.
