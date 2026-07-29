# NUC deploy reconciliation — 2026-07-29

Point-in-time comparison of the NUC (`go2w-nuc`, user `yusenzlabnuc`,
host `yusenzlabnuc-NUC13ANKi7`) runtime deploy surface against this repo at
`main` = `a4bbb79`. This branch is a **capture + report only**: it does not
merge to `main` and does not touch any current repo file. Merge decisions are
left to a human.

Almost the entire deploy surface is byte-identical to the repo (most NUC files
carry mtime `Jul 29 09:42`, i.e. a fresh redeploy). Only **two** genuine gaps
exist, and **neither is reclaimable NUC-unique drift**:

1. `z-manip-piper-passive-feedback.service` is **repo-newer** — the NUC still
   runs the old, un-hardened unit. This is a *deploy lag*, not drift; nothing
   to pull back.
2. `go2w_sensored.urdf` is **repo-missing in this repo** but byte-identical to
   the git-tracked canonical copy in the sibling `go2W_Sim` repo. Captured here
   only as a point-in-time deploy artifact (see below); its source of truth is
   `go2W_Sim`, not this repo.

There is **no `nuc-newer` content anywhere** and **no repo-missing file with
NUC-unique content**. So the snapshot payload is just the one URDF.

## What this snapshot contains

| path | why captured |
| --- | --- |
| `.local/share/z-mobile-manip/go2w_sensored.urdf` | repo-missing in Z-Mobile-manip; substantial (41653 B). **Byte-identical** to `../go2W_Sim/assets/urdf/go2w_sensored.urdf` (git-tracked there: commit `aba344b`, blob `8e5052bf`). Included as the literal deployed artifact; **not** a claim that this repo should own it — reconciliation action is "confirm go2W_Sim is the source of truth", not "add to Z-Mobile-manip". |

Deploy relative paths are preserved under `deploy-snapshot/nuc-20260729/`
rooted at the NUC `$HOME`.

## Full per-file comparison matrix

Repo counterpart columns use the actual deploy source (the installer
`scripts/runtime/install_go2w_reactive_runtime.sh` renames two `configs/nuc-*-init.py`
files into the on-NUC `__init__.py` stubs, so those are compared against their
real sources, not against `z_manip/*/__init__.py`).

### (a) `~/.local/lib/z-mobile-manip/`  →  `scripts/runtime/`

| NUC file | repo counterpart | verdict |
| --- | --- | --- |
| `go2w_base_lock_publish.py` | `scripts/runtime/go2w_base_lock_publish.py` | identical |
| `go2w_base_lock_publish.sh` | `scripts/runtime/go2w_base_lock_publish.sh` | identical |
| `go2w_base_lock.py` | `scripts/runtime/go2w_base_lock.py` | identical |
| `go2w_reactive_control_nuc.py` | `scripts/runtime/go2w_reactive_control_nuc.py` | identical |
| `go2w_reactive_control_nuc.sh` | `scripts/runtime/go2w_reactive_control_nuc.sh` | identical |
| `piper_reactive_view_executor.py` | `scripts/runtime/piper_reactive_view_executor.py` | identical |
| `piper_reactive_view_executor.sh` | `scripts/runtime/piper_reactive_view_executor.sh` | identical |
| `piper_staged_grasp_executor.py` | `scripts/runtime/piper_staged_grasp_executor.py` | identical (107490 B, the Jul29 current code — **not** the stale 61494 B Jul17 copy under `~/z-manip-runtime`) |

### (a) `~/.local/lib/z-mobile-manip/z_manip/` package

| NUC file | repo counterpart | verdict |
| --- | --- | --- |
| `z_manip/__init__.py` | `z_manip/__init__.py` | identical |
| `z_manip/fixed_self_collision.py` | `z_manip/fixed_self_collision.py` | identical |
| `z_manip/control/go2w_posture.py` | `z_manip/control/go2w_posture.py` | identical |
| `z_manip/control/__init__.py` | **`configs/nuc-control-init.py`** | identical (intentional trimmed deploy stub; installer renames it on NUC) |
| `z_manip/kinematics/chain.py` | `z_manip/kinematics/chain.py` | identical |
| `z_manip/kinematics/__init__.py` | **`configs/nuc-kinematics-init.py`** | identical (intentional trimmed deploy stub; installer renames it on NUC) |

> Note: the on-NUC `__init__.py` stubs deliberately import only what is deployed
> (`chain.py`), not `pinocchio_ik`/`robust_ik`. A naive deploy that copied the
> repo's full `z_manip/kinematics/__init__.py` onto the NUC would crash on import
> there. The stubs are load-bearing; `install_go2w_reactive_runtime.sh` handles
> this correctly via the `configs/nuc-*-init.py` sources.

### (b) `~/.local/lib/z-manip/`

| NUC file | repo counterpart | verdict |
| --- | --- | --- |
| `go2w_visual_servo_base_nuc.sh` | `scripts/runtime/go2w_visual_servo_base_nuc.sh` | identical |
| `go2w_visual_servo_base_nuc.sh.pre-speed-018` | (backup, no repo counterpart) | on-NUC backup only; current `.sh` already == repo, so the speed-018 change is already in the repo |

### (c) `~/.local/share/z-mobile-manip/`

| NUC file | repo counterpart | verdict |
| --- | --- | --- |
| `piper_collision_capsules.json` | `configs/piper_collision_capsules.json` | identical |
| `piper_collision_capsules.json.bak-r0325-20260728` | (backup, no repo counterpart) | on-NUC backup only, older revision |
| `go2w_sensored.urdf` | **none in this repo** | repo-missing → captured (== `go2W_Sim` source) |

### (d) `~/.config/z-mobile-manip/`

| NUC file | repo counterpart | verdict |
| --- | --- | --- |
| `go2w-reactive-live.env` | `configs/go2w-reactive-live.env` | identical |

### (e) `~/.config/systemd/user/` — the 4 target units

| NUC unit | repo counterpart | verdict |
| --- | --- | --- |
| `z-manip-go2w-base-control.service` | `configs/z-manip-go2w-base-control.service` | identical |
| `z-mobile-manip-go2w-reactive-live.service` | `configs/z-mobile-manip-go2w-reactive-live.service` | identical |
| `z-mobile-manip-piper-reactive-view.service` | `configs/z-mobile-manip-piper-reactive-view.service` | identical |
| `z-manip-piper-passive-feedback.service` | `configs/z-manip-piper-passive-feedback.service` | **repo-newer** — see below |

## The one repo-newer gap: `z-manip-piper-passive-feedback.service`

- NUC installed: 488 B / 15 lines, mtime `Jul 17 15:04`, **enabled** (symlinked
  in `default.target.wants`).
- Repo: 2439 B / 47 lines, mtime `Jul 28 19:39`.
- `diff` is `+32 / -0`: the repo unit is a strict superset — NUC unit + the
  `[Unit]`-section rate-limit hardening block (`StartLimitIntervalSec=60`,
  `StartLimitBurst=8`, plus the long `# H5.` rationale comment). The `[Service]`
  section (ExecStart / Environment / Restart) is byte-identical between the two.
- Direction is unambiguous: **repo-newer**. The NUC has zero unique lines. The
  StartLimit hardening (which prevents the passive-feedback bridge from latching
  under the ~8 hand-restarts a pick→hold→place cycle triggers) has **not** been
  redeployed to the NUC. Nothing to pull back; the actionable item (for a human,
  outside this task's no-restart scope) is to redeploy the repo unit.

## Step 4 — `~/z-manip-runtime` references (report only, nothing deleted)

`~/z-manip-runtime` is the deprecated Z-Manipulation-Stack deploy residue
(61494 B Jul17 `piper_staged_grasp_executor.py`, `piper_home_recovery.py`,
`grasp-actions/`, `reverse-home-now/`, etc.). Grep for live references to it:

- systemd user units: **1 reference** —
  `z-manip-piper-passive-feedback.service:10` ExecStart runs
  `%h/z-manip-runtime/piper_passive_joint_state_bridge.py`. This unit **is
  enabled**. The **repo's own** `configs/z-manip-piper-passive-feedback.service`
  ExecStart points at the same `%h/z-manip-runtime/...` path, so the repo has
  not migrated the passive bridge off that path either.
- `~/.local/lib/**` launchers (`.sh`/`.py`): **no references**.
- `~/go2w-nuc/**` bringup: **no references**.

The referenced script `~/z-manip-runtime/piper_passive_joint_state_bridge.py`
(7489 B, Jul28) is **byte-identical** to the repo's
`scripts/runtime/piper_passive_joint_state_bridge.py` — it is current code
merely deployed under the deprecated path.

**Conclusion:** `~/z-manip-runtime` is **not** safe to blanket-clean. Exactly
one file in it (`piper_passive_joint_state_bridge.py`) is load-bearing: an
enabled, current unit runs it. The rest of the directory (stale 61 KB executor,
home-recovery, action dirs) has no current reference found and would be the only
clean-up candidates — but cleanup requires first migrating the passive-bridge
ExecStart to a supported path (a repo change), and is out of this task's scope.

## Step 5 — repo `configs/*.service` vs NUC installed units

Beyond the 4 target units above, all other overlapping-by-name units match:

| repo `configs/*.service` | NUC installed unit | verdict |
| --- | --- | --- |
| `d435i.service` | `d435i.service` | identical |
| `ffs-ir-throttle.service` | `ffs-ir-throttle.service` | identical |
| `foxglove-bridge-read-only.service` | `foxglove-bridge.service` | identical content (filename differs on NUC) |

Repo `configs/*.service` units with **no** installed counterpart on the NUC
(present in repo, not deployed): `z-manip-local-grounding.service`,
`z-manip-planning-workbench.service`, `z-manip-runtime-observer.service`,
`z-mobile-manip-go2w-posture-intent-live.service`,
`z-mobile-manip-go2w-posture-intent-shadow.service`,
`z-mobile-manip-go2w-reactive-shadow.service`.

NUC-installed units with **no** repo `configs/` counterpart (belong to the
`go2w-nuc` bringup, not this repo): `map-color-publisher.service`,
`nav-stack.service`, `virtual-wall-publisher.service`, `webrtc-video.service`,
`sunshine.service`, `codex-update-manager.service`.

Net: for every unit tracked by this repo and installed on the NUC, the only one
where the two sides differ is `z-manip-piper-passive-feedback.service`, and the
repo is newer.

## sha256 manifest (pulled NUC deploy surface, 2026-07-29)

```
ff447c30710b1e7159f7aad8da760c98a88212e33bd794598f66f615d88d206d  .config/systemd/user/z-manip-go2w-base-control.service
13128e2ddf43982b7973f50259c75b463edfd7877a572452ed7be344fc7d3fb6  .config/systemd/user/z-manip-piper-passive-feedback.service
c637d75b879ea5899a58f51a659c34b43064df720a4bc10e9bffd82e1ae4cee5  .config/systemd/user/z-mobile-manip-go2w-reactive-live.service
ea109075ca1aba74cc0b96443fbe357ee103fa0be3d29a333242be166e59fd80  .config/systemd/user/z-mobile-manip-piper-reactive-view.service
8bd61ea2d5e2ada666536de14ab2c464a39848acd24d9fe8c2cff3417607857e  .config/systemd/user/d435i.service
e18c92ac11b0755730982db46d778493c51d243a9c2a2b5e9b73b7e54ffd3a06  .config/systemd/user/ffs-ir-throttle.service
db8852f54edd81f276811a1d331c19802c3308ddca1a496137ae377a6ae3a8c5  .config/systemd/user/foxglove-bridge.service
7315626069b021772b0ee65bc92afb827dd9e23c58b7b3770349c0dd755356db  .config/z-mobile-manip/go2w-reactive-live.env
a17b5f909b5dc218df6ce68ada5c9e50e040a22d6d040f63210777e4868ec36d  .local/lib/z-manip/go2w_visual_servo_base_nuc.sh
712739d25c47d2bfda82f352d3df029ac66b1f0b99c8b20a3e8192a1f4b8475b  .local/lib/z-manip/go2w_visual_servo_base_nuc.sh.pre-speed-018
bd6a2aa2f2223e54a238ff21e500fe681e1b544dc338a79a024509d4c9ab9a92  .local/lib/z-mobile-manip/go2w_base_lock_publish.py
3f3261eaae0f17a949da29baf1166ad8e5649cc8991669fe63f175e271b4d4cb  .local/lib/z-mobile-manip/go2w_base_lock_publish.sh
28e4be5b4f8e938cf2b84389769286bf74e174af5bb3fea794650ede6d1902c2  .local/lib/z-mobile-manip/go2w_base_lock.py
82fe6eb3c63fc3388171b41800b48944461f5babf963f4516056aec97ba7fdbb  .local/lib/z-mobile-manip/go2w_reactive_control_nuc.py
b2b9ac07093a7853b6ac8f3be352028491b691b618e2dc97700a168c734b4c69  .local/lib/z-mobile-manip/go2w_reactive_control_nuc.sh
c530f9fcf9fd62286f0382b7ebfd6bbeb29483b02c2ce07a1594b481f86ccb85  .local/lib/z-mobile-manip/piper_reactive_view_executor.py
ead31aed35a2006a1462ead3979056924c7cddd5ad4a298e90053a5549582ec0  .local/lib/z-mobile-manip/piper_reactive_view_executor.sh
5fac0876dd732493cbca4a0788d146f5345d6561fe46e4a2aa9c7286b9dd0065  .local/lib/z-mobile-manip/piper_staged_grasp_executor.py
83ab2a6ec2210df2d69158e94718d5165ca7835738634a44906f6eb04f0196a5  .local/lib/z-mobile-manip/z_manip/control/go2w_posture.py
f6f2d7dff597c822da10a72bcdd9392da46d4ebe8de9ee3040e09392e4048ea2  .local/lib/z-mobile-manip/z_manip/control/__init__.py
b179521405db67cb07a4ecb6ca6fa14ff7efed3882bb0cf9ba60ae464dbf1976  .local/lib/z-mobile-manip/z_manip/fixed_self_collision.py
f7ad7544069e0690ee856cdfa3495e38164182c6c1bff7703b78112f905b3b76  .local/lib/z-mobile-manip/z_manip/__init__.py
c1485983257d6e53d405440832382d0c59030a1764c7d538762faef2f649d405  .local/lib/z-mobile-manip/z_manip/kinematics/chain.py
0a3ca8e1191f4354fa977fbdde322c10f24477c4b79407ada1a807a3c06b018d  .local/lib/z-mobile-manip/z_manip/kinematics/__init__.py
100b37f19adcd7b79877de8b8e8e06919fd679ea3a6410ffe05dd315b4966134  .local/share/z-mobile-manip/go2w_sensored.urdf
48a5fa3a9135a793e636c2953d25fc458461825976fdad06846a10d36c695496  .local/share/z-mobile-manip/piper_collision_capsules.json
c3ae314c910c59335b9508c89f2185ef9e101e682c100c38d2c3b99af78e41fb  .local/share/z-mobile-manip/piper_collision_capsules.json.bak-r0325-20260728
```
