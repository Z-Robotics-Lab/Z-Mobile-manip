# Z-Manipulation-Stack Plan

> 分层移动抓取栈：Go2W 轮足狗 + 背部 PiPER 6DoF 臂 + 腕部 D435i，先在 Isaac Sim 跑通感知规控，
> 后迁真机（NUC 薄枢纽 + 场外 4090）。z-manip 仓只讲 ROS2 契约，平台差异全进 adapter。
> 本轮 = CEO 审前蓝图，不动代码、不启 sim。

---

## 红队修订记录（本轮审校后的事实校正与门槛补强）

> 以下逐条改动均已回写到对应章节；证据现挂实测/官方原文，替换了此前的乐观表述。

1. **[实证错误·A预算承重] HGGD 许可证 = MIT，非 Apache-2.0**（已核 github.com/THU-VCLab/HGGD 页脚为 MIT license）。
   §0 概览、G2、§4b 对比表、§6 license 行四处更正。MIT 商用同样无碍，但作为 CEO 选型证据必须写准。
   M2T2 = Apache-2.0 经复核属实，保留。
2. **[实证错误·最承重] HGGD「无重 CUDA 依赖 / sm_120 ✅」为假 → 降级为「sm_120 兼容性待实测（风险中高）」**。
   官方 README 仅在 Cuda 11.1/11.3/11.6 + PyTorch 1.10-1.12 + numpy==1.23.5 测过，且硬依赖
   pytorch3d / cupoch / numba / grasp_nms——pytorch3d 无 sm_120 预编 wheel、cupoch 是 CUDA 编译库，
   两者恰是 Blackwell 上最易编不过的。§0 风险①、G2、§4b、§6 三处叙事同步改写；A 预算默认件的
   「一上 5080 即可跑」假设撤销，改为「需在 CUDA12.8 重编 pytorch3d/cupoch，风险中高」。
3. **[实证错误] D435i IMU = BMI085，非 BMI055**（Intel PCN 118035-00 已将 D435i IMU 由 BMI055 换为 BMI085）。
   G8、§4 相机(真机)行更正为 BMI085（新批次）。
4. **[gate 可测性] M0-M3 每个里程碑补一条机器可判定的「进入下一阶段」准则**（§5 表新增列 + §3 状态机 gate 数值化）。
   肉眼信号降为附加信号，不再是唯一门槛。
5. **[属主冲突] 补 servo_base 近段与 pathFollower 的 /cmd_vel 接管时序**（G7 展开 + §3 APPROACH 阶段）。
   现状 agent_bridge nav_owner 只管 /way_point 属主、不管 /cmd_vel，两个生产者会打架，故明写抢占/静默/交回时序。
6. **[缺失项] 补 PLACE 阶段与全局重试/中止预算表**（§3 状态机新增 PLACE、SEARCH not_found 判据、每技能重试上限 N）。
7. **[臂-步态耦合 gate] 补 base pitch/roll 阈值 gate 与展臂质心扰动的在线监控/收臂触发**（§2 铁则展开 + §3 ALIGN/GRASP gate）。

---

## 0. 一页概览（CEO 30 秒）

| 项 | 内容 |
|----|------|
| **目标** | 自然语言「把 X 拿来」→ 狗自己找物、走到 standoff、抓起、递送。plan·route·verify·recover 一条链。 |
| **路线** | M0 臂三姿态+腕相机出流 → M1 find+两段伺服进近+追踪 → M2 抓取位姿生成 → M3 规划执行(肉眼见拿起) → M4 严格 verify harness+E2E+red-team → M5 真机契约冻结。**M0-M3 只留免费调试信号，绝不宣称完成/验收。** |
| **最大风险** | ① 5080 16G 与 Isaac 共存显存账（抓取推理挤占，**待实测**）；② RTF 0.2 把任何墙钟超时放大 5×（伺服环假失败）；③ sm_120/Blackwell 需 CUDA 12.8+：GraspGen 高精度 backbone 官方明说跑不了，**且 A 预算默认件 HGGD 亦硬依赖 pytorch3d/cupoch（无 sm_120 预编 wheel，需 CUDA12.8 重编，风险中高——sm_120 兼容性待实测）**；④ PiPER 626mm 臂展边界 IK 病态。 |
| **要拍板的事** | 见 §1 决策队列 8 项：新 repo 创建、抓取生成选型(A/B 双预算)、检测/追踪/规划器/IK 求解器新外部依赖、agent_bridge 契约扩展(新 owner+新端点)、真机硬件接口冻结。 |
| **本轮性质** | 只读蓝图。凡涉新 msg/srv、cross-package 数据流、新外部依赖、硬件接口 = CEO gate，待签，不自越。 |

---

## 1. 决策队列（CEO gates）

每项 = 选项 → 推荐 → 理由 → 证据。

| # | 决策 | 选项 A | 选项 B | 推荐 | 理由 + 证据 |
|---|------|--------|--------|------|------------|
| **G1** | 新建 z-manip repo | 独立新 repo（薄编排 + refs gitignored + 补丁集） | 塞进 go2w 或 z-agent | **A** | 沿 go2w 已验证模式（refs/ 只读、补丁集、clone_deps.sh）。z-manip 只消费 ROS2 topic，与 Isaac 发布代码（落 go2w）解耦，符合宪法 invariant 3/4「一个通用 driver、kernel 不 import world」。**新 repo 创建 = CEO gate。** |
| **G2** | 抓取生成 A 预算（sim 开发期，与 Isaac 共享 5080 16G） | HGGD（端到端单视角 RGBD，实时，MIT license；**sm_120 兼容性待实测**——硬依赖 pytorch3d/cupoch/numba/grasp_nms，需 CUDA12.8 重编，风险中高） | Contact-GraspNet PyTorch 移植（权重自带，输入 depth+K+分割） | **A（HGGD 主，几何 antipodal 兜底）** | GraspGen 的 PTv3 高精度 backbone 官方明写「不能在 CUDA 12.8/Blackwell 跑」，A 预算排除它当默认。HGGD 显存 <8-10GB 可与 office sim 共存，但**官方 README 只在 Cuda 11.1/11.3/11.6 + PyTorch 1.10-1.12 + numpy==1.23.5 测过，且依赖 pytorch3d（无 sm_120 预编 wheel）/cupoch（CUDA 编译库）/numba/grasp_nms——这些恰是 Blackwell 上最易编不过的**，故 HGGD 上 5080 前必须先实测能否在 CUDA12.8 重编成功，不作「即插即用」假设。永远并行保留纯几何 antipodal 基线（零依赖、CPU、A/B 锚点），HGGD 编不过时它是 A 预算实际默认件。证据: github.com/THU-VCLab/HGGD(MIT + requirements.txt) · github.com/NVlabs/GraspGen(README sm_120 坑) · arxiv 2504.19716(QuickGrasp 几何) |
| **G3** | 抓取生成 B 预算（真机，场外 4090 独占 24G） | GraspGen 全量 PTv3 | M2T2（Apache-2.0，点云 transformer，可语言条件） | **A（GraspGen 主，M2T2 若 license 顾虑则替）** | **CEO 2026-07-09 已定：先本机（A 预算）做，4090 后议——本项整体后置，不阻塞 M0-M3；sim/real 通用不受影响（L0 adapter 契约不变，B 预算只是把 L1 模型进程换台机器跑）。** 4090 上 CUDA 12.8 装齐可跑全量 diffusion + cuRobo 并行过滤。M2T2 是唯一明确 Apache-2.0 的 NVIDIA 抓取件，无商用障碍，作升级/交叉校验。证据: graspgen.github.io · huggingface wentao-yuan/m2t2 · arxiv 2311.00926 |
| **G4** | 开放词表检测 + 3D 追踪 | YOLO-E(检测/分割) + EdgeTAM(3D track) + VLM 消歧（同事栈同款） | 自建/其它 | **A** | 同事两仓已趟通，provider-agnostic 抽象保留。抄其 track_3d 契约。**新外部依赖 = CEO gate。** 证据: R2 AGENTS.md:30-44 · MANIPULATION_STACK_SETUP.md:263-275。**VLM 供应商已定（CEO 2026-07-10）：走 OpenRouter，弃 Qwen 直连 API**。同一 Isaac 腕相机货架帧复测后，两路均保留 40 s 独立上限；2026-07-15 现场延迟/成功率证据将 `qwen/qwen3-vl-235b-a22b-instruct` 调为主模型、`qwen/qwen3.5-35b-a3b` 后备。仅 typed transient transport failure 重试一次；32B 没有通过进近方向、禁碰区域和完整目标框门槛。完整盲测见 `docs/vlm_benchmark_2026-07-10.md`。配置面：`OPENROUTER_API_KEY` + `Z_MANIP_VLM_MODEL`（见 .env.example），密钥 gitignored 手工携带 |
| **G5** | 运动规划器 | MoveIt2 + OMPL RRTConnect（CEO 已指定起步） | VAMP(RRTConnect，同事用) | **A 起步，B 后置升级** | CEO 拍板 RRT 先行，VAMP 是后续项。VAMP 需为每机型 codegen（yam.hh 头 + foam 球化），前期成本高。证据: R1 motion_planner.py:87 · R2 §9 VAMP 配方 |
| **G6** | IK 求解器 | TRAC-IK(SQP，抗关节限位) | pick_ik(bio_ik 后继，Jazzy 原生，local/global 双档) | **A 主 + B 备**（近奇异用 DLS 兜底） | 两者 Jazzy 均可。TRAC-IK **不支持 mimic joint**——PiPER 平行爪联动指须排除出 IK 链，只解 6 臂关节。证据: github.com/aprotyas/trac_ik · docs.ros.org/jazzy/pick_ik |
| **G7** | agent_bridge 契约扩展 | 加新端点(/arm_pose, /camera；/grasp 已有) + nav_owner 加 `manip_servo` 态 | 不扩，另起新桥 | **A** | 复用现有 nav_owner 互斥机制。新 owner=manip_servo。**关键补漏（属主冲突）**：现状 nav_owner **只管 /way_point 属主，不管 /cmd_vel**（R3 agent_bridge.py:35-58），而 pathFollower 仍在持续产 /cmd_vel——故 manip_servo 近段直控 /cmd_vel 时必须显式接管，见下方【近段 /cmd_vel 接管时序】。**cross-package 数据流 + 契约扩展 = CEO gate。** 证据: R3 agent_bridge.py:35-58,251-262 |
| **G8** | 真机硬件接口冻结（M5） | D435i(realsense2_camera Jazzy) + piper_sdk(CAN) + zenoh 桥 | — | **A** | **硬件接口 = CEO gate，M5 才冻结。** IMU 可做手眼校核；**注意 Intel PCN 118035-00 已将 D435i IMU 由 BMI055 换为 BMI085**，新批次为 BMI085——冻结前按实机批次核对 IMU 型号（影响标定与 driver 配置）。跨 WiFi 走 zenoh-bridge-ros2dds。证据: dev.realsenseai.com/docs/ros2-wrapper · Intel PCN 118035-00 · github.com/eclipse-zenoh/zenoh-plugin-ros2dds |

**【近段 /cmd_vel 接管时序】**（补 G7/§2/§3——两个 /cmd_vel 生产者的抢占/静默/交回契约）：

| 步 | 触发 | manip_servo 动作 | pathFollower 侧效果 | 交回 |
|----|------|-----------------|--------------------|------|
| 1 抢占 | base 进 <1.5m 近段、nav_owner 置 `manip_servo` | 撤销当前 /way_point（发空/远段目标清零），使 pathFollower 收敛到「无路点→零速空转」；同时 manip_servo 独占发 /cmd_vel | pathFollower 因无 /way_point 输出零速，不再竞争 /cmd_vel（推荐首选：撤路点静默，不停节点，保留其存活可随时交回） | — |
| 2 保持 | 近段伺服进行中 | manip_servo 持续发 /cmd_vel（低速+upright gate）；心跳看门狗（sim-dt 计），丢心跳→零速 | pathFollower 保持零速空转 | — |
| 3 释放 | GRASP 序列开始（base 停稳）或 RECOVER | manip_servo 停发 /cmd_vel、发一帧零速定住，nav_owner 释放回 `nav`/空 | 重新发 /way_point 即恢复 pathFollower 正常产 /cmd_vel | base 控制权交回 nav |

> 单一真源原则：任一时刻 /cmd_vel 只有一个逻辑生产者，由 nav_owner 态机排他保证；**绝不同时让 pathFollower 与 manip_servo 都产 /cmd_vel**。若后续要更硬的隔离，升级为 twist_mux 优先级（manip_servo > pathFollower），但起步用「撤路点使 pathFollower 空转」即可，零新节点。对照 R3 agent_bridge.py:35-58 现有 owner 语义。

---

## 2. 分层架构（L0-L4）

| 层 | 职责 | sim/real 契约边界 |
|----|------|------------------|
| **L4 zeno 任务层** | 自然语言→技能编排(plan·route·verify·recover)，经 z-agent go2w world 注册 seam（tools/vocab/persona） | 纯 Python，经 agent_bridge HTTP :8042。平台无关。注册路径见 R3 go2w.py:643-653,814-913 |
| **L3 技能层** | find(X)/approach(X)/align(X)/pick(X)/carry/place(X)。每技能 = 进入条件 + 动作 + 确定性 verify predicate + 超时/重试/降级 | 只发 ROS2 action/topic，不含算法（抄同事 L5→L3 分层，R2 modularity.md:8-48） |
| **L2 原语层** | scan（原地旋转/臂扫视+逐帧检测）、servo_base（两段式）、arm_goto（named pose/关节轨迹）、track（EdgeTAM 掩码流）、grasp_exec（pre-grasp→直线进近→闭爪→提升） | ROS2 action 契约。servo_base 近段直控 cmd_vel（新 owner=manip_servo）|
| **L1 模型层** | Detector(open-vocab 2D)、VLM(消歧)、Tracker、GraspSource(GraspGen/HGGD 主 · AnyGrasp 备 · GT-heuristic 仅 bring-up)、Planner(MoveIt2-RRT 基线 / VAMP 升级) | **全藏在可替换接口后**。抄同事 HardwareBackend Protocol(typing.Protocol 零耦合)，R2 modularity.md:181-208 |
| **L0 适配层** | isaac_adapter（腕部 RGBD + TF + 关节接口 + GT verify hooks）· real_adapter（D435i + piper_sdk） | **Isaac 发布代码落 go2w 仓，z-manip 只消费 ROS2 topic，绝不 import Isaac。** 平台差异全进此层，迁真机=只改 DDS 配置 |

**铁则**：base 与 arm 运动初期互斥——导航=STOW 收臂；伺服进近=LOOKOUT 但低速+upright gate；停稳才展开抓取序列。loco-manip（边走边操作）是后期里程碑不是起点。

**臂-步态耦合 gate（补——展臂质心扰动的在线监控）**：LOOKOUT 展臂使质心前移/上移，扰动步态策略（model_5495 已训入背臂+载荷包络，但 LOOKOUT 姿态属其分布边缘）。故 ALIGN/GRASP 进入前与全程守一条 **base 姿态 gate**：读 base pitch/roll（来源：IMU `/imu` 或里程计 `/state_estimation` 的姿态四元数，M5 真机同源），**|pitch|、|roll| 超阈（起步阈 pitch ±12°、roll ±10°，M0 标定期实测收紧）即判姿态失稳 → 臂立即回 STOW 并暂停伺服/抓取，退回 ALIGN 重稳**。该 gate 是 §3 ALIGN/GRASP 阶段进入条件的一部分，且全程周期性复查（sim-dt 计），不是一次性检查。

---

## 3. 技能状态机

SEARCH→APPROACH→ALIGN→GRASP→VERIFY→CARRY→**PLACE**→RECOVER。超时**一律按 sim 时间 /clock 计，禁墙钟**（RTF 0.2 放大 5×，R3 pitfalls 坑41）。

**进入下一阶段的门槛 = 机器可判定的「gate（M0-3 免费信号也须过此判据）」列；「肉眼信号」列仅作附加确认，不是唯一门槛。**（回应 gate 可测性红队项——M0-M3 用免费信号、不宣称验收，但「完成/进入下一阶段」必须脚本可判。）

| 阶段 | 进入条件 | 动作 | **进入下一阶段 gate（M0-M3 机器可判定）** | 肉眼附加信号 / M4 严格 GT | 超时(sim-s) | 降级 |
|------|---------|------|------------------------------------------|--------------------------|------------|------|
| **SEARCH** | 收到 find(X) | 臂 STOW→scan(原地旋转+臂扫视 LOOKOUT)+逐帧 Detector→VLM 消歧 | 单帧检测置信 ≥τ_det（起步 0.4）且同一 track_id 命中持续 ≥K 帧（K=3），产出稳定 3D pose | RViz 见框+日志 / M4: Detection3D 命中且置信>阈 | 30 | 扩大扫描角度→仍无→**报 not_found**：**回 zeno 让 LLM 重规划（换目标别名/换搜索区）；重规划已达全局上限（见预算表 SEARCH=2）则终止上报「未找到 X」** |
| **APPROACH** | 目标 3D pose 已知 | servo_base 两段式：远段发 /way_point 借 localPlanner 避障；近段 <1.5m **按【近段 /cmd_vel 接管时序】撤路点静默 pathFollower 后**低速直控 /cmd_vel + 地形 gate | base-target 平面距离误差 <standoff_tol（0.10m）**持续 ≥4 sim-s**；或 STUCK 兜底判 base-target ≤success_radius(0.5m) | 距离误差曲线 / M4: base 进 standoff 窗持续 4s | 60 | STUCK→改判 success_radius(0.5m)→重规划一次→放弃 |
| **ALIGN** | 进 standoff **且 base 姿态 gate 过（\|pitch\|≤12°、\|roll\|≤10°，读 /imu 或 /state_estimation）** | 臂 LOOKOUT，EdgeTAM 锁掩码，微调 base yaw 对准（法向/朝物）；守 D435i pre-grasp 视点 **≥0.35m**；**展臂后全程复查姿态 gate，超阈→臂回 STOW+退 ALIGN** | 掩码 IoU 帧间连续 ≥τ_mask(0.5) 持续 ≥K sim-s；base yaw 对目标法向夹角 <φ_yaw(15°)；pre-grasp 深度点有效率 >τ_depth；姿态 gate 持续过 | 掩码肉眼锁住 / M4: 目标在灵巧核心区+深度有效 | 20 | 掩码丢→回 SEARCH；深度空洞→后退加距；姿态失稳→STOW 退 ALIGN |
| **GRASP** | 对齐稳定 **且 base 姿态 gate 过** | GraspSource 出候选→对称扩 SE(3)→IK/规划过滤→MoveIt2-RRT 规划 pre-grasp→Cartesian 直线进近→闭爪→提升；**全程复查姿态 gate，超阈→中止收臂回 STOW** | 候选位姿 approach 轴与物体表面法向夹角 <θ_app(30°) 且 IK 有解且规划成功；闭爪后爪宽 ∈(0,max)；提升 Δz 达标 | 爪宽日志+肉眼拿起 / M4: 爪宽∈(0,max)+提升后物跟随+接触 GT | 40 | IK 无解→base 重定位换 standoff；规划失败→换候选；闭爪空→重试（各计入预算表上限）；姿态失稳→STOW 退 ALIGN |
| **VERIFY** | 提升完成 | 读 GT（sim）/ 爪宽+腕力+视觉三票（真机） | 爪宽 ∈(0,max) 且提升后目标随爪（sim 用 /piper GT oracle；免费信号用「爪宽>0 且深度上目标仍在爪前」代理） | **M0-3: 无验收，仅记免费信号 / M4: sim GT predicate（actor 摸不到 GT）** | 5 | 判失败→RECOVER |
| **CARRY** | verify 通过 | 臂 CARRY 姿态（收于胸前握持位），导航到目的地；**CARRY 姿态载荷偏心亦守姿态 gate** | base 到目的地 <goal_tol 且全程爪宽保持 >0（未掉） | 肉眼未掉 / M4: 全程握持 GT 保持 | 按路径 | 中途掉落（爪宽→0 或深度失踪）→回 SEARCH 重抓 |
| **PLACE** | 到目的地 | 规划 place 位姿（目的地上方 standoff→Cartesian 下放至释放高度）→松爪→抬臂脱手→回 STOW | 释放高度到位；松爪后爪宽→open；抬臂后目标**不随爪**（sim GT 用目标 odom 与爪距 >脱手阈；免费信号用「爪前深度不再有目标」）→判脱手成功 | 肉眼见物体放下且爪松开 / M4: 目标静置目的地 + 已脱手 GT | 20 | 下放受阻/未脱手→抬臂重试（计入预算表 PLACE 上限）；耗尽→RECOVER 报「未能放置」 |
| **RECOVER** | 任一阶段失败 | 臂回 STOW，清状态，按技能重试/降级策略决定重试或上报 | — | — | — | 超该技能重试上限（见预算表）→上报 zeno 让 LLM 重规划或终止 |

### 3a. 全局重试 / 中止预算表（每技能重试上限 N 与耗尽后行为）

> 单技能内失败先按该阶段「降级」列自愈；累计达下表 N 即停止该技能自愈，交回上层（zeno/LLM 重规划）或终止。避免任一阶段无界重试卡死整链。

| 阶段/失败类型 | 单次自愈动作 | 重试上限 N | 耗尽后行为 |
|--------------|-------------|:---------:|-----------|
| SEARCH not_found | 扩大扫描角度 / 换别名重扫 | 2 | 终止，回 zeno 上报「未找到 X」 |
| APPROACH STUCK | 重规划路径一次 | 1 | 放弃本目标，回 zeno |
| ALIGN 掩码丢失 | 回 SEARCH 重锁 | 2 | 回 zeno 重规划（可能目标已移动/遮挡）|
| GRASP IK 无解 | base 重定位换 standoff | 3 | 回 zeno（该物在此位姿不可抓）|
| GRASP 规划失败 | 换下一抓取候选 | 5（候选数上限）| IK/候选均尽→base 重定位（回 GRASP IK 计数）|
| GRASP 闭爪空 | 原位重试闭爪 | 2 | 回 GRASP 换候选 |
| PLACE 未脱手 | 抬臂重放一次 | 2 | RECOVER 报「未能放置」，物仍在爪，回 zeno |
| **全链 pick 尝试**（SEARCH→VERIFY 一整轮）| 整轮重来 | 3 | 终止任务，回 zeno 上报「无法完成，把 X 拿来」 |

### 3b. 视觉伺服设计（合并小节——三个环，只有前两个闭环）

| 环 | 传感 / 误差量 | 控制量 | 频率 | 闭环？ |
|----|--------------|--------|------|:------:|
| 远段 APPROACH（>1.5m） | SLAM 定位 + 目标 3D pose → 平面距离误差 | /way_point（借 localPlanner 避障） | 按需重发 | 半闭环（导航栈内自闭环） |
| 近段 APPROACH + ALIGN（<1.5m） | EdgeTAM 掩码质心 + D435i 深度 → (距离误差, yaw 夹角) | /cmd_vel 低速直控（按 G7 接管时序） | **4-10Hz，sim-time 计** | **闭环（position-based servo）** |
| GRASP 下扎（最后 ~30cm） | 无——D435i min-Z 0.28m 盲区 | Cartesian 直线 min-jerk（开环） | 一次规划 | **开环 + 爪开口余量兜底** |

**为什么臂端不做经典 IBVS/PBVS 高频闭环（有意取舍，非遗漏）**：

1. **物理限制**：D435i min-Z 0.28m——最后一段深度必然失效，图像闭环闭不到接触点；闭环收益集中在 pre-grasp 之前，那段用 4Hz track 重规划已覆盖；
2. **拓扑限制**：真机图像→场外算力→控制回传跨 WiFi，RTT 使 >10Hz 视觉闭环不可靠；「硬实时不跨 WiFi」铁则下高频视觉闭环无处安放；
3. **sim 限制**：RTF 0.2 把墙钟环放大 5×，高频闭环在 sim 里只会更抖；
4. **同行验证**：同事栈同款取舍已在真机跑通（R1：4Hz 重规划 + PoseGoalSmoother(alpha=0.6, deadband=0.01) + 开环直线 sting；task_planner.md:183 明确把 visual servoing 划出范围）。

**误差平滑与到位判据**（抄现成）：PoseGoalSmoother 平滑目标 + 收敛契约「stop_update_distance 窗内持续 convergence_duration + STUCK 兜底 success_radius」（R2 visual_servoing_base/README.md:92-118）。

**升级路径（M4 后可选，不进 M0-M3）**：PBVS 细化段——pre-grasp 到位后用掩码质心+深度做 1-2 次离散位姿修正再下扎；真机实测 WiFi RTT 稳 <30ms 时近段闭环可提至 10Hz。

---

## 4. 技术选型表

| 组件 | 推荐 | 备选 | 理由 | 证据 URL / file:line | CEO gate |
|------|------|------|------|---------------------|:--------:|
| open-vocab 检测 | YOLO-E | Grounding-DINO 等 | 同事已用，无 SAM2 依赖 | R2 AGENTS.md:31,232 | ✅ |
| 3D 追踪 | EdgeTAM | — | 同事 track_3d 已趟通 | R2 MANIPULATION_STACK_SETUP.md:266 | ✅ |
| VLM 消歧 | **OpenRouter `qwen/qwen3-vl-235b-a22b-instruct`** | `qwen/qwen3.5-35b-a3b` | 两者通过货架几何门槛；现场调用中 235B 延迟/成功率更稳定，32B 不进入生产回退链 | `docs/vlm_benchmark_2026-07-10.md` | ✅已批 |
| 运动规划 | MoveIt2 + OMPL RRTConnect | VAMP(RRTConnect) | CEO 拍板 RRT 起步，VAMP 后置 | R1 motion_planner.py:87 | ✅ |
| 抓取执行 | Cartesian 直线下扎(min-jerk) | — | 抄同事，明确不含接触检测（重力/PD 误触发） | R1 cartesian_trajectory_controller.py:97-330 | — |
| 跨机桥 | zenoh-bridge-ros2dds | CycloneDDS unicast-only | Jazzy 有，治 DDS 发现洪泛 | github.com/eclipse-zenoh/zenoh-plugin-ros2dds | ✅ |
| 相机(真机) | realsense2_camera(Jazzy) | — | 官方支持，含 IMU（**新批次 BMI085**，Intel PCN 118035-00 由 BMI055 更换；按实机批次核对） | dev.realsenseai.com/docs/ros2-wrapper · Intel PCN 118035-00 | ✅ |
| 相机(sim) | Isaac 通用 Camera prim 改内参对齐 D435i | Isaac D455 USD 改参 | 无原生 D435 asset；需加噪+裁近距防虚假成功 | docs.isaacsim.omniverse.nvidia.com/latest/sensors | — |

### 4a. IK 近极限策略（PiPER 626mm 臂展，狗背抓远物）

移动底盘是**第一 IK 资源**；硬解边界 IK 是兜底不是常态。四段管线：

| 段 | 手段 | 证据 URL |
|----|------|---------|
| ① 候选可达性/可操作度过滤 | Reuleaux 逆可达图选 base standoff，让目标落进 PiPER 灵巧核心区（远离 626mm 边界）；cuRobo GPU batch-IK 一次过滤可达+无碰子集（每秒 3.7 万解，比 TRAC-IK 快 23-80×，**属场外算力绝不上狗背 NUC**） | wiki.ros.org/reuleaux · curobo.org/reports/curobo_report.pdf |
| ② 平行爪绕 approach 轴姿态松弛 | 每抓取候选枚举绕 approach 轴 N 采样 + 180° yaw 翻转（两指对调等价），放大成一族 SE(3) 目标，显著提可解率 | arxiv 2504.19502 · rss13/p15(relaxed-rigidity) |
| ③ 求解器 | TRAC-IK(SQP，抗关节限位，**排除爪指 mimic joint 只解 6 臂关节**) 主 / pick_ik(local 做 Cartesian、global 救远初值) 备 | github.com/aprotyas/trac_ik · docs.ros.org/jazzy/pick_ik |
| ④ 近奇异/边界兜底 | 变阻尼 DLS（Chiaverini 自适应 / SDLS 按 SVD 只压奇异分量，远奇异不白丢精度）；仍无解→**base 重定位换 standoff** | Springer BF01254007 · arxiv 2604.13405 |

PiPER 官方参数（AgileX）：臂展 ~626mm（PiPER-X 669mm）、载荷 1.5kg、重复精度 ±0.1mm、6 关节 CAN。证据: global.agilex.ai/products/piper

### 4b. 抓取生成对比表（GraspGen vs 轻量替代，A=sim/B=真机 双预算）

| 组件 | 单视角契约 | 显存 | 延迟 | license | sm_120 | 预算归属 | 证据 URL |
|------|-----------|------|------|---------|--------|:--------:|---------|
| **HGGD** | 端到端单视角 RGBD | <8-10GB | 实时(定性，精确 ms **待实测**) | **MIT** | ⚠️ **sm_120 待实测**（依赖 pytorch3d 无预编 wheel + cupoch CUDA 编译库 + numba/grasp_nms，官方仅测 Cuda11.1/11.3/11.6，需 CUDA12.8 重编，风险中高） | **A 首选(编过为前提)** | github.com/THU-VCLab/HGGD |
| **Contact-GraspNet(PyTorch)** | depth+K(+可选 2D 分割) | ≥8GB | 0.19-0.28s(原 TF) | 移植非官方 | 需 CUDA12.8 重编 op | A 备选 | github.com/elchun/contact_graspnet_pytorch |
| **几何 antipodal(QuickGrasp 式)** | 点云 PCA 对趾+力闭合 | 0(纯 CPU) | ms 级 | 零依赖 | ✅ | **A/B 永久兜底+锚点** | arxiv 2504.19716 |
| **GraspGen** | 点云 diffusion | PTv3 需 CUDA12.1；PointNet++ 变体可上 5080 | ~20Hz 推理 | NVIDIA 专有研究 | ❌ PTv3 官方明说不跑 Blackwell | **B 主力(4090)** | github.com/NVlabs/GraspGen |
| **M2T2** | 点云 transformer(可语言条件) | A100 ~121ms 级 | — | **Apache-2.0** | 4090 可 | B 升级/交叉校验 | huggingface wentao-yuan/m2t2 |
| **AnyGrasp/GSNet** | 点云几何(不吃 RGB) | 16G 级(同事跑 RTX3080) | — | 商业 license(MAC 机器锁) | MinkowskiEngine 编译坑 | 备选(需申请 license) | R1 anygrasp README:9-46 |

**A 预算推荐**：HGGD 主 + 几何 antipodal 兜底；显存均 <10GB 可与 office sim 共享 5080。**但 HGGD 上 5080 有前置门槛——先实测其 pytorch3d/cupoch 能否在 CUDA12.8/sm_120 重编成功**（官方仅测 Cuda11.1/11.3/11.6，pytorch3d 无 sm_120 预编 wheel、cupoch 是 CUDA 编译库，风险中高）；编不过则 A 预算实际默认件退回纯几何 antipodal（CPU、零依赖）。**不选 GraspGen 当 A 默认**（PTv3 不上 Blackwell + 专有 license + 预训练权重无 PiPER 平行爪）。
**B 预算推荐**：GraspGen 全量 PTv3（4090 CUDA12.8 装齐）或 M2T2（Apache-2.0），几何基线仍作 fallback。
**未实测数值（待 clone 实测，不作 load-bearing）**：GraspGen 精确参数量/checkpoint 大小、HGGD 精确单帧 ms、diffusion 采样步数、**HGGD 在 CUDA12.8/sm_120 的可编译性（最关键待测项）**。

---

## 5. 里程碑 M0-M5

**M0-M3 只留免费调试信号（RViz 可视化/爪宽与位姿日志/距离误差曲线/肉眼 rollout），绝不宣称完成/验收。M4 起 = 严格验收 gate。**
**但「进入下一里程碑」需可脚本判定**：下表「进入下一里程碑 gate（机器可判定）」列是硬门槛，「肉眼附加信号」列仅作附加确认，不是唯一门槛。

| 里程碑 | 范围 | **进入下一里程碑 gate（机器可判定）** | 肉眼附加信号 | 预估轮数 | 依赖 |
|--------|------|--------------------------------------|-------------|:--------:|------|
| **M0** | 仓库 bootstrap（G1）+ 臂三姿态(STOW/LOOKOUT/CARRY，落 go2w) + 腕部 D435i RGBD 出流 | `ros2 topic hz /camera/color` ≥ N fps（起步 N=10）且 depth 同频；相机 optical frame 相对 base 的 pitch 在 **±X°**（起步 X=5，平视）由 TF 数值核对；三姿态到位关节误差 <ε | RViz 看到腕相机画面、相机平视前方 | 3-5 | go2w piper_grasp.py + warehouse_nav 相机已发布(R3:819-837) |
| **M1** | find(X) + SCAN + 两段伺服进近 + EdgeTAM 追踪 | base 进 standoff 窗内 base-target 平面距离误差 <standoff_tol(0.10m) **持续 ≥K sim-s**（K=4）；EdgeTAM 掩码帧间 IoU ≥τ_mask(0.5) 连续 ≥K sim-s（无丢失重锁）| 狗自己走到目标物前站进 standoff、掩码一路锁住 | 6-10 | M0 + G4(检测/追踪) + G7(nav_owner manip_servo + /cmd_vel 接管时序) |
| **M2** | 抓取位姿生成管道（GT-heuristic/几何 antipodal 先通管道 → HGGD 按 R5 接入，**HGGD 以 CUDA12.8 重编通过为前提**）+ RViz 候选可视化 | 每帧产出 ≥1 候选，其 approach 轴与物体表面法向夹角 <θ_app(30°)；候选位姿投影回 base frame 数值一致（frame 对齐验证过，非肉眼）| 候选位姿画在物体上、朝向合理 | 5-8 | M1 + G2(HGGD/几何) + 点云 frame 对齐（先做数值验证）|
| **M3** | 规划+执行：MoveIt2-RRT 起步 + IK 近极限三层策略（§4a）+ base 姿态 gate（§2）| GRASP 阶段 IK 有解且规划成功、闭爪后爪宽 ∈(0,max)、提升后 Δz 达标且爪宽保持 >0（免费信号代理「拿起」）| 肉眼见爪子把物体从桌上拿起 | 8-12 | M2 + G5(MoveIt2) + G6(TRAC-IK) |
| **M4** | verify harness 严格化 + zeno E2E「把 X 拿来」+ 失败恢复 + red-team + 成功率统计入账 | sim GT predicate 通过（actor 摸不到 GT）；GT-heuristic 抓取此后禁用，验收一律用学习型生成 | — | 8-15 | M3 + docs/VERIFY.md 风格判据 |
| **M5** | real adapter 契约冻结（D435i/piper_sdk/标定 + WiFi 桥实测带宽延迟）+ 真机干跑清单 | 契约冻结、桥带宽/延迟实测达标、干跑清单过 | — | 6-10 | M4 + G8(硬件 gate 签字，含 IMU 实机批次核对) |

---

## 6. 风险表

| 风险 | 诚实算术 / 现状 | 缓解 |
|------|----------------|------|
| **5080 16G 与 Isaac 共存显存** | sim 空闲 GPU 已用 955MiB/16303。office sim 运行时占用**未实测（待实测）**。HGGD/CGN 抓取推理 <8-10GB，理论可共存但叠加 Isaac render 后余量**待实测**。实测法：bringup 后 `nvidia-smi --query-gpu=memory.used -l 5` 采峰值 + `docker exec go2w-isaac ps -eo rss,comm --sort=-rss` 看 kit-python(R3) | 抓取推理先用几何 antipodal(零显存)通管道；HGGD 接入前先单独实测 sim 运行时余量；重模型(GraspGen)一律 B 预算场外 4090 |
| **RTF 0.2 伺服时序** | 墙钟 10s 才走 2 sim-s。抓取阶段超时 _TIMEOUTS 按 sim-s(R3 piper_grasp.py:31)在墙钟=50/40/10/40s。pick 全链 240s 墙钟预算(R3 go2w.py:557) | 所有伺服/抓取环超时**按 sim 时间 /clock 或 sim-dt 计，禁墙钟**（同 R3 坑41 导航死锁根因）|
| **sm_120/Blackwell 兼容** | RTX 5080=sm_120，最低 CUDA 12.8，首个原生 wheel=PyTorch 2.7.0。带自定义 CUDA op 的网络(graspnet-baseline 锁 PyTorch1.6、Contact-GraspNet、GPD)需 CUDA12.8 重编，很多仓库几乎编不过。**⚠️ 修正：A 预算默认件 HGGD 并非「无重 CUDA 依赖」——它硬依赖 pytorch3d（无 sm_120 预编 wheel）/cupoch（CUDA 编译库）/numba/grasp_nms，官方仅在 Cuda11.1/11.3/11.6 + PyTorch1.10-1.12 测过，这些恰是 Blackwell 上最易编不过的。「一上 5080 即跑」的假设已撤。** | **HGGD 上 5080 前先做重编可行性实测（pytorch3d/cupoch 在 CUDA12.8/sm_120），编不过则退回纯几何 antipodal（CPU、零依赖）作 A 预算实际默认**；重 op 网络若必须用则场外 4090 单独环境；证据 forums.developer.nvidia.com Blackwell 迁移指南 · github.com/THU-VCLab/HGGD requirements.txt |
| **license** | GraspGen=NVIDIA 专有研究 license（商用需联系）；AnyGrasp=商业 MAC 机器锁；预训练权重均 Franka/Robotiq 无 PiPER。**修正：HGGD=MIT（非 Apache-2.0），M2T2=Apache-2.0（已核）——二者商用均无碍** | 主力用 **MIT 的 HGGD / Apache-2.0 的 M2T2**；几何 antipodal 完全无 license；AnyGrasp 仅在申请到 license 后备选 |
| **PiPER 控制保真** | Isaac 侧 piper_joint1-8 已配执行器(R3 warehouse_nav.py:218-243)；TRAC-IK 不支持 mimic joint，平行爪联动指须排除 | IK 只解 6 臂关节，爪宽单独控；真机 CAN 400ms 看门狗自动进 damping（R1 i2rt_real_setup.md 同类经验）|
| **office 场景可抓物** | 唯一道具 /World/GraspBox=6cm 红箱硬编码(R3 warehouse_nav.py:185-197)，office USD(酒店大堂)无原生可抓道具 | 多物需扩 SCENES box→boxes + BOX_CFG 参数化 + /objects/<name>/odom 多路（**新话题结构=CEO gate**）；改 box 位置是 launch 期烘焙，配对重启 bringup |
| **WiFi 桥带宽/延迟** | 见 §9 算术：848×480 RGBD 裸流 488Mbps 打不过 WiFi；H264+RVL @30fps≈72Mbps、@15fps≈36Mbps。depth 只能无损(RVL ~3:1) 是瓶颈 | 15fps + 裁 ROI + 变化触发按帧上传 depth；硬实时环绝不跨 WiFi |

---

## 7. 仓库骨架（沿 go2w 模式）

```
~/Desktop/z-manip/
├── AGENTS.md                    # 宪法（若跑自演化循环）
├── docs/
│   ├── plan.md                  # 本文件
│   ├── VERIFY.md                # M4 才写：sim GT predicate 判据
│   └── WIRING.md                # 子系统布线（新增/改行为时更新）
├── z_manip/                     # 薄编排（colcon 包 / pip 包）
│   ├── skills/                  # L3: find/approach/align/pick/carry/place
│   ├── primitives/              # L2: scan/servo_base/arm_goto/track/grasp_exec
│   ├── models/                  # L1: 全藏接口后
│   │   ├── grasp_source.py      #   GraspSource Protocol（HGGD/AnyGrasp/GT-heuristic 后端）
│   │   ├── detector.py          #   open-vocab 2D
│   │   ├── tracker.py           #   EdgeTAM 掩码流
│   │   └── planner.py           #   MoveIt2-RRT / VAMP
│   ├── ik/                      #   IK 四段管线（对称扩→过滤→求解→DLS 兜底）
│   └── adapters/                # L0: isaac_adapter(只消费 topic) / real_adapter(D435i+piper_sdk)
├── refs/                        # gitignored 只读参考仓（clone_deps.sh 拉）
│   ├── vector_manipulation_stack/
│   └── vector_robotics/
├── patches/                     # 对 refs 的补丁集（不改 refs 原文件）
├── scripts/
│   └── clone_deps.sh            # 拉 refs + 抓取推理服务器镜像
└── docker/                      # 抓取推理 ZMQ 服务器（照抄同事 docker/{graspgen,anygrasp}）
```

**铁则**：refs/ gitignored 纯只读；对参考仓的改动进 patches/；Isaac 发布代码落 go2w 仓不落此仓。

---

## 8. 参考仓借鉴清单

**抄什么（file 级指针）**：

| 借鉴点 | 来源 file:line |
|--------|---------------|
| backend 级联 + 统一 GraspGenerator 契约（generate(ctx)→candidates，空则降级几何） | R1 grasp/base.py:32-140 · grasp/registry.py:52-141 |
| pregrasp 锥形采样(纯几何零 IK)+下游权威校验分工 | R1 grasp/pregrasp_sampler.py:1-255 |
| gripper-mesh 点云碰撞过滤(KDTree+min_contact_points，容忍指尖擦桌) | R1 gripper_transforms.py:44-355 |
| HardwareBackend Protocol(typing.Protocol 零耦合)+Pinocchio 通用 IK | R2 modularity.md:181-208 · R1 arm_interface.py:71-92 |
| 五层分层契约(Skill→Gadget→Feature→Utility，层间只走 ROS msg) | R2 modularity.md:8-48 |
| ZMQ 抓取服务器骨架(msgpack-over-ZMQ REP，health/metadata/infer) | R2 docker/anygrasp/anygrasp_server.py:1-90 |
| 边/云 Zenoh 桥拓扑(robot 压缩流→host GPU→结果回传，共享 ROS_DOMAIN_ID) | R2 remote_bridge/README.md:80-105 |
| 到位收敛契约(stop_update_distance 窗持续 convergence_duration，STUCK 兜底 success_radius) | R2 visual_servoing_base/README.md:92-118 |
| bag_replay 离线回放(latched 单帧重发驱动抓取反复触发) | R2 bag_replay/README.md:1-64 |
| 新臂接 VAMP 全配方(URDF→SRDF→foam→cricket) | R2 MANIPULATION_STACK_SETUP.md:386-621 |
| 自家现成件零改动复用：PiperGraspController 顶抓状态机、/grasp 端点、holding_object GT oracle、Go2WArm/Gripper 鸭子合同、/piper/* 五路 GT | R3 piper_grasp.py 全文 · agent_bridge.py:251-262 · go2w.py:424-486,790-800 · warehouse_nav.py:734-775 |
| 新技能进 planner 正门(strategies/descriptions/params_help/examples 四处一致) | R3 go2w.py:814-913 |

**不抄什么（一句理由）**：

| 不抄 | 理由 | 来源 |
|------|------|------|
| 同事整栈的抓取成败验证 | 闭爪即成功，无提升/夹持宽度/接触检测——直接违反我方宪法 invariant 1「verify 是护城河」，必须自补 | R1 grasp_object.py:737-742 |
| 「视觉伺服/移动底盘协同=范围外」的定位 | 同事是固定基座桌面臂；我方 Go2W 移动底盘+背臂协同恰是核心，其设计无参考甚至误导 | R1 task_planner.md:183 |
| MuJoCo 仿真后端 | 我方是 Isaac Sim 5.1，需写 IsaacBackend；MuJoCoBackend 不可直接复用 | R2 §5 |
| mecanum 底盘运动学假设(lateral_offset 横移) | 同事全向麦轮，我方轮足横移弱，servo 对位假设需重估 | R2 §5 |
| GraspGen 当 sim 默认 | PTv3 不上 Blackwell + 专有 license + 无 PiPER 权重 | R5 GraspGen README |
| graspnet-baseline | 锁 PyTorch1.6 + 需编 pointnet2/knn op，sm_120 几乎编不过，非商用 | R5 graspnet-baseline README |

---

## 9. 部署拓扑（sim→real bridge）

### 真机算力分工

```
┌─────────────────────────────┐        WiFi (同网, ROS2/DDS 桥)       ┌──────────────────────────────┐
│  狗背 NUC (i7/16G) 薄 I/O 枢纽 │  ── 压缩 RGBD 下行 (H264+RVL) ──▶   │  场外笔记本 / 4090 主机          │
│  · 导航 (CMU 栈 /cmd_vel)     │                                     │  · 感知 (YOLO-E/EdgeTAM/VLM)   │
│  · D435i 发布                │  ◀── 目标位姿 / 轨迹上行 ──────       │  · 抓取生成 (GraspGen/HGGD)    │
│  · PiPER CAN 执行            │                                     │  · 运动规划 (MoveIt2/cuRobo)   │
│  【硬实时环闭在此机】          │                                     │  【全部重算力】                 │
│  · 步态 RL 在狗上闭环         │                                     │                              │
│  · 臂关节插值在 PiPER SDK/固件 │                                     │                              │
└─────────────────────────────┘                                     └──────────────────────────────┘
```

### 跨 WiFi 话题清单与带宽算术（848×480，自算确定性）

| 方向 | 话题 | 编码 | 裸流 | 压缩后 @30fps | @15fps |
|------|------|------|------|--------------|--------|
| 下行 robot→host | /camera/color | H264(~40:1) | 293Mbps | ~7.3Mbps | ~3.7Mbps |
| 下行 robot→host | /camera/depth | RVL(~3:1，无损，H264 会毁深度) | 195Mbps | ~65Mbps | ~33Mbps |
| **下行合计** | — | H264+RVL | **488Mbps(打不过 WiFi)** | **~72Mbps** | **~36Mbps** |
| 上行 host→robot | /grasp_pose, /trajectory, Detection3D | 小 msg | — | <1Mbps | <1Mbps |

**结论**：depth 是带宽瓶颈（RVL 仅 ~3:1）。共享 WiFi 强烈建议 **15fps + 裁 ROI + 变化触发按帧上传 depth**。证据: 自算 + image_transport compressed_depth(RVL) + nvidia-isaac-ros compression。

### 延迟预算表（伺服环频率上限 vs WiFi RTT）

| 环 | 位置 | 频率上限 | 跨 WiFi？ | 说明 |
|----|------|---------|:--------:|------|
| 步态 RL locomotion | 狗上闭环 | 50-100Hz | **否** | 硬实时，绝不跨 WiFi |
| 臂关节插值 | PiPER SDK/固件 | 100-250Hz | **否** | 硬实时，闭在固件 |
| 视觉伺服(base 对准) | 跨 WiFi 慢环 | ≤4-10Hz | 是 | 受 WiFi RTT 限；4Hz 重规划（抄同事 R1 track_object 4Hz）|
| 抓取生成 | 场外 GPU | 按需单次 | 是 | 非实时，一次触发 |

**铁则**：任何硬实时环（步态/臂插值）绝不跨 WiFi；跨 WiFi 只走压缩 RGBD 下行 + 目标位姿/轨迹上行。

### sim 从 M0 起如何镜像该拓扑

- manip 栈 = **独立进程**，只经 ROS2 topic 与 Isaac 通信，**不共享内存**（迁真机=只改 DDS 网络配置）。
- Isaac 侧发布代码落 go2w 仓；z-manip 只订阅 /camera/*、/piper/*、TF。
- sim 全链 use_sim_time=true；桥须正确透传 /clock（同事 bag_replay 未做此项，我方要补，R2）。
- sim 深度需**人为加噪+裁近距(<0.3m 打空洞)** 逼近 D435i，防虚假成功迁真机翻车。

### 迁移=只改 DDS 配置的验证清单

1. z-manip 全程无 `import isaac`/无 Isaac API 调用（grep 核）。
2. 所有平台差异隔离在 L0 adapter（isaac_adapter ↔ real_adapter 可替换）。
3. topic 名 + 编码在 sim/real 一致（16UC1 depth / rgb8，对齐 realsense2_camera）。
4. 切换只动：DDS profile(CycloneDDS unicast XML 或 zenoh 桥配置) + adapter 选择参数。
5. 坐标系每一跳先写 5 行数值验证（q=0 两条 FK 链 4×4 位姿差，0° 才直用）——抓取失败 90% 是 frame 坑（R2 MISTAKES.md #5/#9/#11）。
