from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "web/debug_dashboard/runtime_scene.js"


def _node(script: str) -> dict[str, object]:
    result = subprocess.run(
        ["node", "-e", script, str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _harness(assertions: str) -> str:
    assertions = assertions.replace("{{", "{").replace("}}", "}")
    return rf"""
const fs = require('fs');
const vm = require('vm');
global.window = global;
global.devicePixelRatio = 1;
global.HTMLCanvasElement = function() {{}};
global.matchMedia = () => ({{ matches: true }});
const operations = [];
const context = new Proxy({{ operations }}, {{
  get(target, key) {{
    if (key in target) return target[key];
    return (...args) => operations.push([String(key), ...args]);
  }},
  set(target, key, value) {{ target[key] = value; return true; }}
}});
class Canvas {{
  constructor() {{
    this.width = 640;
    this.height = 480;
    this.rect = {{ width: 640, height: 480 }};
    this.listeners = new Map();
  }}
  getContext(kind) {{ return kind === '2d' ? context : null; }}
  getBoundingClientRect() {{ return this.rect; }}
  addEventListener(name, callback) {{ this.listeners.set(name, callback); }}
  removeEventListener(name) {{ this.listeners.delete(name); }}
  setPointerCapture() {{}}
  emit(name, values = {{}}) {{
    const callback = this.listeners.get(name);
    if (callback) callback(Object.assign({{
      clientX: 0, clientY: 0, button: 0, pointerId: 1,
      shiftKey: false, deltaY: 0, preventDefault() {{}}
    }}, values));
  }}
}}
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'), {{ filename: process.argv[1] }});
const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const cameraPose = [[1,0,0,.2],[0,1,0,-.1],[0,0,1,.4],[0,0,0,1]];
const graspPose = [[1,0,0,.35],[0,1,0,.05],[0,0,1,.25],[0,0,0,1]];
const pregraspPose = [[1,0,0,.25],[0,1,0,.05],[0,0,1,.25],[0,0,0,1]];
const links = [
  [[0,0,0],[0,0,.12]],
  [[0,0,.12],[.18,0,.22]],
  [[.18,0,.22],[.32,.02,.28]]
];
const witness = {{
  frame: 'piper_base_link', kind: 'scene', radius_m: .04,
  capsule_start_base: [.16,0,.2], capsule_end_base: [.3,.02,.27],
  witness_scene_point_base: [.27,.06,.27], witness_capsule_point_base: [.27,.02,.27]
}};
const bundle = {{
  schema: 'z_manip.debug_bundle.v1',
  frames: {{ perception: 'camera', planning: 'piper_base_link' }},
  visualization: {{
    frame: 'piper_base_link', robot_overlay_allowed: true,
    scene_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.2,.1,0],[.3,-.1,.02]] }},
    target_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.32,.03,.21]] }},
    reference_axes: [
      {{ name: 'base', frame: 'piper_base_link', pose: identity }},
      {{ name: 'camera', frame: 'piper_base_link', pose: cameraPose }}
    ],
    candidate_axes: [{{ candidate_id: 3, rank: 1, status: 'selected', frame: 'piper_base_link', pose: graspPose }}],
    robot_overlay: {{ frame: 'piper_base_link', links_xyz_m: links, joint_positions_rad: [0,0,0,0,0,0] }},
    planned_robot_overlay: {{ frame: 'piper_base_link', links_xyz_m: links.map(pair => pair.map(p => [p[0],p[1]+.02,p[2]])) }},
    trajectory_xyz_m: {{ transit: [[0,0,.12],[.18,0,.22]], approach: [[.18,0,.22],[.32,.03,.25]], lift: [[.32,.03,.25],[.32,.03,.35]] }}
  }},
  selected_plan: {{ candidate_id: 3, pregrasp_pose_base: pregraspPose, grasp_pose_base: graspPose }},
  planning: {{ rejections: [{{ candidate_index: 3, collision_witness: witness }}] }}
}};
{assertions}
"""


def test_module_exposes_explicit_dependency_free_api_and_safety_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "root.ZManipScene = API",
        "create(canvas, options)",
        "setBundle(bundle)",
        "setJointState(value)",
        "setPlannedState(value)",
        "setSelection(value)",
        "resetView()",
        "destroy()",
        "prefers-reduced-motion: reduce",
        "maxFps: 30",
        "FRAME_MISMATCH",
        "OVERLAY_LOCKED",
        "_drawFloor",
        "_drawCamera",
        "_drawGripper",
        "_drawCollision",
    ):
        assert required in source
    for forbidden in (
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "EventSource",
        "create_publisher",
        ".publish(",
        "socketcan",
        "cansend",
        "/cmd_vel",
        "/joint_trajectory",
        "innerHTML",
    ):
        assert forbidden not in source


def test_renders_full_frame_consistent_scene_and_reduced_motion_updates_immediately():
    result = _node(_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{ autoResize: false, interactive: true }});
const accepted = scene.setBundle(bundle);
const before = scene.getState().renderCount;
const updateAccepted = scene.setJointState({{
  frame: 'piper_base_link', links_xyz_m: links,
  joint_positions_rad: [.01,.02,.03,.04,.05,.06]
}});
const after = scene.getState();
canvas.emit('pointerdown', {{clientX: 10, clientY: 10}});
canvas.emit('pointermove', {{clientX: 35, clientY: 20}});
canvas.emit('pointerup');
canvas.emit('wheel', {{deltaY: -100}});
scene.flush();
const interactive = scene.getState();
canvas.rect = {{width: 800, height: 500}};
const viewport = scene.resize(false);
const output = {{
  api: Object.keys(window.ZManipScene).sort(),
  accepted,
  before,
  after,
  interactive,
  viewport,
  operations: operations.length,
  diagnostics: scene.getDiagnostics()
}};
scene.destroy();
console.log(JSON.stringify(output));
"""))

    assert result["accepted"]["accepted"] is True
    assert result["accepted"]["frame"] == "piper_base_link"
    assert result["after"]["reducedMotion"] is True
    assert result["after"]["maxFps"] == 30
    assert result["after"]["renderCount"] > result["before"]
    assert result["after"]["counts"] == {
        "scenePoints": 2,
        "targetPoints": 1,
        "candidates": 1,
        "actualLinks": 3,
        "plannedLinks": 3,
        "collisionWitness": 1,
        "diagnostics": 0,
    }
    assert result["interactive"]["orbit"]["zoom"] > 1
    assert result["interactive"]["orbit"]["yaw"] != -0.72
    assert result["viewport"] == {"width": 800, "height": 500, "ratio": 1}
    assert result["operations"] > 100
    assert result["diagnostics"] == []
    assert result["api"] == ["VERSION", "create", "isSupported", "validateFrame"]


def test_frame_mismatches_and_locked_overlays_fail_soft_without_replacing_state():
    result = _node(_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
bundle.visualization.target_cloud.frame = 'camera_color_optical_frame';
bundle.visualization.robot_overlay_allowed = false;
const accepted = scene.setBundle(bundle);
const before = scene.getState();
const wrongUpdate = scene.setPlannedState({{frame:'map',links_xyz_m:links}});
const trajectoryUpdate = scene.update({{
  trajectory: {{frame:'map',segments:{{approach:[[0,0,0],[.1,0,0]]}}}}
}});
scene.render();
const after = scene.getState();
console.log(JSON.stringify({{
  accepted, before, wrongUpdate, trajectoryUpdate, after,
  diagnostics: scene.getDiagnostics().map(item => item.code)
}}));
"""))

    assert result["accepted"]["accepted"] is True
    assert result["before"]["counts"]["scenePoints"] == 2
    assert result["before"]["counts"]["targetPoints"] == 0
    assert result["before"]["counts"]["actualLinks"] == 0
    assert result["wrongUpdate"] is False
    assert result["trajectoryUpdate"] is False
    assert result["after"]["counts"]["plannedLinks"] == 0
    assert "FRAME_MISMATCH" in result["diagnostics"]
    assert "OVERLAY_LOCKED" in result["diagnostics"]


def test_non_reduced_updates_are_coalesced_behind_the_30hz_budget():
    result = _node(_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {
  autoResize:false, interactive:false, reducedMotion:false, maxFps:30
});
scene.setBundle(bundle);
const before = scene.getState().renderCount;
const first = scene.setJointState({frame:'piper_base_link',links_xyz_m:links});
const second = scene.setJointState({frame:'piper_base_link',links_xyz_m:links});
const queued = scene.getState();
scene.flush();
const flushed = scene.getState();
scene.destroy();
console.log(JSON.stringify({before,first,second,queued,flushed}));
"""))

    assert result["first"] is True
    assert result["second"] is True
    assert result["queued"]["maxFps"] == 30
    assert result["queued"]["pendingUpdate"] is True
    assert result["queued"]["renderCount"] == result["before"]
    assert result["flushed"]["pendingUpdate"] is False
    assert result["flushed"]["renderCount"] == result["before"] + 1


def test_live_cloud_updates_do_not_move_locked_virtual_camera_and_pose_labels_render():
    result = _node(_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.setBundle(bundle);
const before = scene.getState().framing;
scene.update({{
  sceneCloud: {{frame:'piper_base_link',points_xyz_m:[[-100,-100,-100],[100,100,100]]}},
  targetCloud: {{frame:'piper_base_link',points_xyz_m:[[20,20,20],[21,21,21]]}},
  actualRobot: {{
    frame:'piper_base_link',
    links_xyz_m: links.map(pair => pair.map(value => [value[0] + .01, value[1], value[2]]))
  }}
}});
const after = scene.getState().framing;
const labels = operations
  .filter(operation => operation[0] === 'fillText')
  .map(operation => operation[1]);
console.log(JSON.stringify({{before,after,labels}}));
"""))

    assert result["before"] == result["after"]
    assert result["before"]["source"] == "bundle_locked"
    assert "camera pose" in result["labels"]
    assert "grasp pose #3" in result["labels"]


def test_display_voxels_suppress_sub_5mm_jitter_but_show_real_motion_immediately():
    result = _node(_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {
  autoResize:false, interactive:false, reducedMotion:true, cloudVoxelM:.005
});
scene.setBundle(bundle);
const before = scene.getState().renderCount;
const jitter = scene.update({
  sceneCloud: {
    frame:'piper_base_link',
    points_xyz_m:[[.201,.101,.001],[.301,-.099,.021]]
  }
});
const afterJitter = scene.getState();
const movement = scene.update({
  sceneCloud: {
    frame:'piper_base_link',
    points_xyz_m:[[.210,.100,0],[.300,-.100,.020]]
  }
});
const afterMovement = scene.getState();
console.log(JSON.stringify({before,jitter,afterJitter,movement,afterMovement}));
"""))

    assert result["jitter"] is False
    assert result["afterJitter"]["renderCount"] == result["before"]
    assert result["afterJitter"]["filters"]["displayOnly"] is True
    assert result["afterJitter"]["filters"]["cloudVoxelM"] == 0.005
    assert result["afterJitter"]["filters"]["cloudUpdatesSuppressed"] == 1
    assert result["movement"] is True
    assert result["afterMovement"]["renderCount"] == result["before"] + 1


def test_missing_or_malformed_geometry_is_fail_soft():
    result = _node(_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
const empty = scene.setBundle({{visualization:{{frame:'camera',robot_overlay_allowed:false}}}});
const malformed = scene.update({{
  sceneCloud: {{frame:'camera',points_xyz_m:[[NaN,0,0],['bad',0,0]]}},
  collisionWitness: {{frame:'camera',kind:'scene'}}
}});
const rendered = scene.render();
console.log(JSON.stringify({{empty,malformed,rendered,state:scene.getState(),diagnostics:scene.getDiagnostics()}}));
"""))

    assert result["empty"]["accepted"] is True
    assert result["rendered"] is True
    assert result["state"]["counts"]["scenePoints"] == 0
    codes = {item["code"] for item in result["diagnostics"]}
    assert "BASE_FRAME_UNAVAILABLE" in codes
    assert "INVALID_POINTS" in codes
    assert "INVALID_COLLISION_WITNESS" in codes


def test_failed_plan_does_not_auto_select_rejection_witness_and_explicit_selection_draws_pair():
    result = _node(_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
bundle.selected_plan = null;
bundle.planning.plan_valid = false;
bundle.visualization.candidate_axes[0].status = 'rejected';
scene.setBundle(bundle);
const before = scene.getState();
const beforeLabels = operations
  .filter(operation => operation[0] === 'fillText')
  .map(operation => operation[1]);
scene.setSelection({{candidateId:null,rejection:bundle.planning.rejections[0]}});
const after = scene.getState();
const afterLabels = operations
  .filter(operation => operation[0] === 'fillText')
  .map(operation => operation[1]);
console.log(JSON.stringify({{before,beforeLabels,after,afterLabels}}));
"""))

    assert result["before"]["selectedCandidateId"] is None
    assert result["before"]["counts"]["collisionWitness"] == 0
    assert "scene nearest" not in result["beforeLabels"]
    assert result["after"]["selectedCandidateId"] is None
    assert result["after"]["counts"]["collisionWitness"] == 1
    assert "scene nearest" in result["afterLabels"]
    assert "capsule nearest" in result["afterLabels"]


def test_javascript_syntax_is_valid():
    subprocess.run(["node", "--check", str(SCRIPT)], check=True)
