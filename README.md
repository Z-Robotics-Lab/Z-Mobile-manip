# Z-Mobile-Manip

Z-Mobile-Manip 是一套面向研究与现场演示的监督式移动抓取系统，硬件组合为
Unitree Go2-W EDU、AgileX PiPER 6DoF 机械臂、腕部 Intel RealSense D435、机器人侧
NUC 和 RTX 4090 工作站。系统将开放词汇 RGB-D 感知、EdgeTAM 跟踪、深度视觉伺服、
几何抓取候选、Pinocchio IK、碰撞感知规划和有界执行集成到同一个本机 Web 工作台。

当前真机已跑通两条主链路：

- 固定底盘：`perception → planning → grasp → return Home`
- 移动抓取：`find/track → depth approach → stop → close-range grasp`

这是需要操作员在场的研究系统，不是无人值守产品。任何运动测试都必须留出净空并确保
实体急停可触达。

## 系统结构

| 位置 | 主要职责 |
|---|---|
| 4090 PC | RGB-D 解码、目标检测/跟踪、点云、抓取候选、IK/规划、UI |
| Go2-W NUC | D435 ROS 服务、PiPER `can0`、被动关节反馈、短生命周期执行器、底盘 WebRTC 速度控制 |
| Browser | 感知、规划、执行、Home/reset、Full Stop 和诊断 |

PC 与 NUC 默认使用同一 Wi-Fi 和 `ROS_DOMAIN_ID=20`。主流程为：

```text
目标文本 → RGB-D grounding → EdgeTAM tracking → 目标点云
→ 粗对齐并以深度闭环接近 → 约 0.50 m 停车 → 近场重新感知
→ 抓取候选 → Pinocchio IK → 碰撞检查与路径规划
→ pregrasp → approach → slow close → smooth lift → Home
```

机械臂在离开 Home 前规划完整路径；进入 D435 近距离盲区后不依赖再次感知或重新规划。

## 硬件与软件要求

- Ubuntu 24.04、ROS 2 Jazzy、CycloneDDS
- NVIDIA GPU、Docker Engine、NVIDIA Container Toolkit
- Unitree Go2-W EDU 与可通过 SSH 访问的机载 NUC
- AgileX PiPER、1 Mbps SocketCAN 接口和正确的机器人 URDF
- RealSense D435/D435i（使用彩色与对齐深度，不要求 IMU）
- PC/NUC 时间同步，ROS Domain ID 一致

实际底盘传输使用第三方 `unitree_webrtc_connect`；PiPER 执行后端使用 `pyAgxArm`。
EdgeTAM、Pinocchio 和 RealSense/ROS 依赖由运行环境提供。AnyGrasp 是可选后端，其 SDK、
license 与权重不包含在本仓库中。第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 首次配置

```bash
git clone https://github.com/Z-Robotics-Lab/Z-Mobile-manip.git
cd Z-Mobile-manip
cp .env.example .env
install -Dm755 scripts/runtime/manip ~/.local/bin/manip
```

编辑 `.env`，至少确认：

- `ROS_DOMAIN_ID=20`
- `GO2W_NUC_HOST` 与 `GO2W_NUC_SSH_KEY`
- 真实 URDF、手眼标定和安装外参路径
- 相机、CAN 与可选模型服务配置

`configs/piper_home.example.json` 仅是文件格式示例，**不能作为真机 Home 执行目标**。
每台机器人必须通过被动关节反馈采集并审核自己的 `configs/piper_home.json`；该文件及真实
标定数据默认不会被 Git 跟踪。

详细安装、标定与 NUC 服务配置见
[运行手册](docs/go2w_piper_operations.md) 和
[配置说明](docs/configuration.md)。

## 每日启动

两台机器开机后，在 4090 PC 上运行：

```bash
manip bringup
manip status
manip url
```

工作台默认地址是 <http://127.0.0.1:8766/>。`bringup` 启动 UI、相机桥、被动反馈、
observer、EdgeTAM 与 perception；`status` 会分别报告 NUC 相机、RGB-D、跟踪和反馈状态。

常用维护命令：

```bash
manip component restart nuc-camera
manip component restart perception-all
manip logs perception-all 100
manip restart
manip stop
```

## UI 工作流

固定底盘抓取：

1. 点击 **Reset + Recheck Home**，确认 Home、反馈和旧任务已清理。
2. 输入目标物体，例如 `白色充电器`，运行 **Perception**。
3. 检查 mask、目标点云和抓取候选，运行 **Planning**。
4. 确认现场安全后执行 **Direct Perform**。

移动抓取：

1. 输入目标物体，选择 **Find → Approach → Grasp**。
2. 系统检测或搜索目标，粗略对齐后以 RGB-D 深度闭环前进。
3. 进入近场交接距离后锁定底盘，重新感知、规划并抓取。
4. 任何时刻可点击 **Full Stop** 中断视觉伺服并发送零速度。

UI 中的候选、路径与诊断均来自当前 session；演示前可使用 **Clear Demo** 清除展示数据。
完整状态机、阈值、故障恢复和日志位置见
[运行手册](docs/go2w_piper_operations.md)。

## 安全边界

- UI 能驱动底盘、机械臂和夹爪；只有标为 shadow/planning-only 的操作才不发送运动命令。
- UI 默认只绑定 `127.0.0.1`，但它不是身份认证系统，不应直接暴露到局域网或公网。
- 执行前检查 D435 视野、底盘路径、机械臂扫掠空间、CAN 状态和实体急停。
- 不在无人值守时运行真机执行；不要绕过固件急停或物理限位。
- `.env`、SSH key、真实 Home/标定、rosbag、现场图像、日志和商业模型 license 不得提交。

更多安全与凭据报告说明见 [SECURITY.md](SECURITY.md)。

## 验证

不连接执行器的静态与单元测试：

```bash
python3 -m compileall -q z_manip scripts/runtime ros2
pytest -q
```

需要已有 ROS/真机链路的测试会按环境条件 skip；测试不会替代现场运动验收。

## 项目布局

```text
z_manip/         感知、抓取、IK、规划与运行时 Python 模块
scripts/runtime/ manip CLI、bringup、诊断、视觉伺服与执行入口
web/             本机调试与操作工作台
ros2/            ROS 2 bridge、observer 与接口包
configs/         可公开的 schema 与示例配置
docker/          4090 推理/规划运行环境
tests/           单元、契约与回归测试
docs/            操作、标定、配置和验收文档
```

## 文档

- [真机运行与故障恢复](docs/go2w_piper_operations.md)
- [组件管理器与一键 bringup](docs/component_manager.md)
- [配置 schema 与迁移](docs/configuration.md)
- [安装外参与运动学校准](docs/piper_mount_and_kinematic_calibration.md)
- [移动抓取验收](docs/mobile-manipulation-acceptance.md)
- [分阶段抓取契约](docs/staged_pick_hold_contract.md)
- [架构蓝图与路线](docs/plan.md)

## License

本项目代码采用 [Apache License 2.0](LICENSE)。第三方软件、模型、机器人 SDK 和商标仍受
各自许可证与权利声明约束。本项目与 Unitree Robotics、AgileX Robotics、Intel、NVIDIA
及其关联公司不存在隶属、赞助或背书关系。
