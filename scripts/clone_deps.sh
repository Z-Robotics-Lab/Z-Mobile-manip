#!/usr/bin/env bash
# 幂等拉取 z-manip 的只读参考仓到 refs/（gitignored，绝不入库）。
# 参考 ~/Desktop/go2w/scripts/clone_deps.sh 风格：--filter=blob:none 省带宽，可重复运行。
#
# 参考仓（借鉴清单见 docs/plan.md §8；契约抄用见 z_manip/ 各模块 docstring）：
#   - VectorRobotics/vector_manipulation_stack  抓取 backend 契约（GraspContext/GraspCandidates/
#                                               GraspGenerator）、ZMQ 抓取服务器骨架、五层分层。
#   - VectorRobotics/vector_robotics (-b alexl_svd_dev)  视觉伺服收敛契约、track_3d、docker/{anygrasp,graspgen}。
#
# 铁则（docs/plan.md §7）：refs/ 纯只读；对参考仓的任何改动进 patches/，绝不直改 refs 原文件。
# 抓取推理镜像（docker/{graspgen,anygrasp}）后续 M2 接入时在此拉取，M0 只备结构。
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p refs

# vector_manipulation_stack —— 抓取 backend 契约与 ZMQ 服务器骨架来源。
if [ ! -d refs/vector_manipulation_stack ]; then
  git clone --filter=blob:none \
    https://github.com/VectorRobotics/vector_manipulation_stack.git \
    refs/vector_manipulation_stack
else
  echo "refs/vector_manipulation_stack 已存在，跳过 clone（如需更新自行 git -C 拉取）"
fi

# vector_robotics —— 视觉伺服/track_3d/docker 抓取服务器来源；固定到 alexl_svd_dev 分支。
if [ ! -d refs/vector_robotics ]; then
  git clone --filter=blob:none --branch alexl_svd_dev \
    https://github.com/VectorRobotics/vector_robotics.git \
    refs/vector_robotics
else
  echo "refs/vector_robotics 已存在，跳过 clone（如需更新自行 git -C 拉取）"
fi

# 重新套用 patches/ 下对 refs 的补丁（若有；幂等）。refs 是 gitignored，每次全新 clone 后需重打。
if [ -d patches ] && ls patches/*.patch >/dev/null 2>&1; then
  for p in patches/*.patch; do
    echo "套用补丁 $p"
    git -C refs apply --check "../$p" 2>/dev/null && git -C refs apply "../$p" \
      || echo "  跳过（已套用或不适用）：$p"
  done
fi

echo "OK: refs/vector_manipulation_stack + refs/vector_robotics(alexl_svd_dev)"
echo "抓取推理镜像（docker/graspgen · docker/anygrasp）在 M2 接入时拉取。"
