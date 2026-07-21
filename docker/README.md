# docker/ —— 抓取推理 ZMQ 服务器（占位，M2 接入）

这里未来放**抓取位姿生成模型的推理服务**，以独立 Docker 镜像跑，与 z-manip 主进程解耦。
服务器进程内导入重 CUDA 抓取模型（HGGD / AnyGrasp / GraspGen 等，依预算与 sm_120 兼容性而定，
见 [docs/plan.md](../docs/plan.md) §4b），对外只暴露一个 **msgpack-over-ZMQ REP** 接口
（`{"action": "health" | "metadata" | "infer"}`，`infer` 请求携 `points(N,3)` / `colors(N,3)|None`
/ `lims` / `dense` 等，回 `grasps(M,4,4)` / `scores` / `widths`）。z-manip 侧的 L1
`GraspSource` 后端只做这个 ZMQ 客户端，不在主进程装抓取模型的重依赖——契约与骨架照抄参考仓
`refs/vector_robotics/docker/{anygrasp,graspgen}`（见 docs/plan.md §8 借鉴清单）。

把推理关进镜像是有意的隔离：抓取模型的 CUDA/torch 版本（sm_120 重编、numpy 锁版）与主栈
彻底解耦；真机部署时该服务跑在场外 4090（B 预算），sim 开发期与 Isaac 共享 5080（A 预算），
z-manip 只改 ZMQ 端点地址，不动代码（docs/plan.md §9 部署拓扑）。

**M0 状态**：本目录仅此说明占位，未落 Dockerfile / 服务器代码。M2 抓取管道接入时，
在此新建 `graspgen/` · `anygrasp/`（各含 Dockerfile + `launch_server.py` + `README.md`），
镜像与服务器代码由 `scripts/clone_deps.sh` 拉取的参考仓骨架改写而来。
