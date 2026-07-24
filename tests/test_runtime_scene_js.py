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
        "coloredCloudPoints": 0,
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


def _live_harness(assertions: str, *, with_document: bool = True) -> str:
    assertions = assertions.replace("{{", "{").replace("}}", "}")
    document = (
        r"""
global.document = {
  createElement(tag) {
    if (tag !== 'canvas') return {};
    const canvas = { width: 0, height: 0 };
    canvas.getContext = () => ({
      createImageData(width, height) { return { data: new Uint8ClampedArray(width * height * 4) }; },
      putImageData() {}
    });
    return canvas;
  }
};
"""
        if with_document
        else ""
    )
    return rf"""
const fs = require('fs');
const vm = require('vm');
global.window = global;
global.devicePixelRatio = 1;
global.HTMLCanvasElement = function() {{}};
global.matchMedia = () => ({{ matches: true }});
{document}
const operations = [];
const context = new Proxy({{ operations }}, {{
  get(target, key) {{
    if (key in target) return target[key];
    return (...args) => operations.push([String(key), ...args]);
  }},
  set(target, key, value) {{ target[key] = value; return true; }}
}});
class Canvas {{
  constructor() {{ this.width = 640; this.height = 480; this.rect = {{ width: 640, height: 480 }}; this.listeners = new Map(); }}
  getContext(kind) {{ return kind === '2d' ? context : null; }}
  getBoundingClientRect() {{ return this.rect; }}
  addEventListener(name, callback) {{ this.listeners.set(name, callback); }}
  removeEventListener(name) {{ this.listeners.delete(name); }}
  setPointerCapture() {{}}
}}
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'), {{ filename: process.argv[1] }});
const cameraPose = [[0,0,1,.1],[-1,0,0,0],[0,-1,0,.31],[0,0,0,1]];
const links = [
  [[0,0,0],[0,0,.12]],
  [[0,0,.12],[.18,0,.22]],
  [[.18,0,.22],[.06,0,.23]]
];
const cloudXyz = new Float32Array([.1,0,.3, .12,.02,.31, .2,-.05,.28, .05,.05,.35]);
const cloudRgb = new Uint8Array([255,0,0, 0,255,0, 0,0,255, 200,200,200]);
{assertions}
"""


def test_live_mode_renders_base_frame_skeleton_camera_and_colored_cloud():
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
const entered = scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
const cam = scene.setLiveCameraPose(cameraPose);
const robot = scene.setLiveRobot({{
  frame:'piper_base_link', links_xyz_m: links, joint_positions_rad:[.01,.02,.03,.04,.05,0]
}});
const cloud = scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
scene.flush();
const state = scene.getState();
const labels = operations.filter(op => op[0] === 'fillText').map(op => op[1]);
const drewCloud = operations.some(op => op[0] === 'drawImage');
console.log(JSON.stringify({{
  entered, cam, robot, cloud, state, labels, drewCloud,
  orbit: state.orbit,
  diagnostics: scene.getDiagnostics().map(item => item.code)
}}));
"""))

    assert result["entered"]["accepted"] is True
    assert result["entered"]["frame"] == "piper_base_link"
    assert result["entered"]["live"] is True
    assert result["entered"]["cloudExpected"] is True
    assert result["cam"] is True
    assert result["robot"] is True
    assert result["cloud"] == 4
    assert result["state"]["live"] is True
    assert result["state"]["frame"] == "piper_base_link"
    assert result["state"]["overlayAllowed"] is True
    assert result["state"]["counts"]["actualLinks"] == 3
    assert result["state"]["counts"]["coloredCloudPoints"] == 4
    assert result["state"]["framing"]["source"] == "live"
    assert result["drewCloud"] is True
    assert "camera pose" in result["labels"]
    assert "base" in result["labels"]
    assert result["diagnostics"] == []
    # Live view snaps to the standing-behind default orbit, distinct from the
    # session default so recorded evidence still renders at its original angle.
    assert result["orbit"]["yaw"] == -0.8
    assert result["orbit"]["pitch"] == -0.38


def test_live_locked_shows_base_skeleton_but_withholds_the_colored_cloud():
    # Hand-eye unverified: the arm skeleton is honest base-frame forward
    # kinematics and still renders, but the colored cloud (which needs the
    # measured transform) is withheld from the shared base scene so a camera-frame
    # cloud can never be co-drawn on the base grid.  The cloud stays in the tile.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:false}});
const robot = scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
const cam = scene.setLiveCameraPose(null);
const cloud = scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
scene.flush();
const state = scene.getState();
const drewCloud = operations.some(op => op[0] === 'drawImage');
console.log(JSON.stringify({{
  robot, cam, cloud, state, drewCloud,
  diagnostics: scene.getDiagnostics().map(item => item.code)
}}));
"""))

    assert result["robot"] is True
    assert result["cam"] is False
    assert result["cloud"] == 0
    assert result["state"]["overlayAllowed"] is True
    assert result["state"]["counts"]["actualLinks"] == 3
    assert result["state"]["counts"]["coloredCloudPoints"] == 0
    assert result["state"]["framing"]["source"] == "live"
    assert result["drewCloud"] is False
    assert "CLOUD_FUSION_LOCKED" in result["diagnostics"]
    assert "OVERLAY_LOCKED" not in result["diagnostics"]


def test_live_observer_offline_locks_everything_without_a_skeleton():
    # No fresh link frames: skeleton unavailable and cloud withheld, so only the
    # grid remains and the model reports no overlays.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:false, cloudExpected:false}});
const robot = scene.setLiveRobot(null);
const cloud = scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
scene.flush();
const state = scene.getState();
const drewCloud = operations.some(op => op[0] === 'drawImage');
console.log(JSON.stringify({{robot, cloud, state, drewCloud}}));
"""))

    assert result["robot"] is False
    assert result["cloud"] == 0
    assert result["state"]["overlayAllowed"] is False
    assert result["state"]["counts"]["actualLinks"] == 0
    assert result["state"]["counts"]["coloredCloudPoints"] == 0
    assert result["drewCloud"] is False


def test_live_mode_is_idempotent_and_preserves_cloud_across_repeated_entry():
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
const again = scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
const kept = scene.getState().counts.coloredCloudPoints;
const reset = scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:false, cloudExpected:false}});
const cleared = scene.getState().counts.coloredCloudPoints;
console.log(JSON.stringify({{again, kept, reset, cleared}}));
"""))

    assert result["again"]["live"] is True
    assert result["kept"] == 4
    assert result["reset"]["live"] is True
    # A gate drop (hand-eye vanishing with the suspended feedback service during
    # a grasp) is a HOLD, not a teardown: the previously fused cloud — verified
    # when it entered the model — stays frozen on stage.  The closed gate only
    # blocks NEW unverified data from being fused (see the locked test above).
    assert result["cleared"] == 4


def test_live_colored_cloud_is_fail_soft_without_a_document():
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
const cloud = scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
const rendered = scene.render();
const drewCloud = operations.some(op => op[0] === 'drawImage');
console.log(JSON.stringify({{
  cloud, rendered, drewCloud,
  points: scene.getState().counts.coloredCloudPoints,
  diagnostics: scene.getDiagnostics().map(item => item.code)
}}));
""", with_document=False))

    assert result["cloud"] == 4
    assert result["rendered"] is True
    assert result["points"] == 4
    assert result["drewCloud"] is False
    assert result["diagnostics"] == []


def test_live_view_is_z_up_so_the_robot_up_axis_renders_straight_up():
    # The turntable projection is Y-up, but the live base frame is Z-up.  A point
    # directly above the base (+z) must render straight up on screen: same x, lower
    # y than the base.  We read the drawn axis labels ("base" ~ base origin,
    # "camera pose" ~ camera origin) which fillText places at each origin.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
// Camera origin exactly 0.4 m straight up (+z) from the base, identity rotation.
scene.setLiveCameraPose([[1,0,0,0],[0,1,0,0],[0,0,1,.4],[0,0,0,1]]);
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
scene.flush();
const texts = operations.filter(op => op[0] === 'fillText');
const base = texts.find(op => op[1] === 'base');
const camera = texts.find(op => op[1] === 'camera pose');
console.log(JSON.stringify({{base, camera}}));
"""))

    assert result["base"] is not None
    assert result["camera"] is not None
    # fillText payload is [text, x, y]; the camera (+z above base) is higher on
    # screen (smaller y) and horizontally aligned with the base.
    base_x, base_y = result["base"][2], result["base"][3]
    cam_x, cam_y = result["camera"][2], result["camera"][3]
    assert cam_y < base_y - 20
    assert abs(cam_x - base_x) < 6


def test_verified_framing_clamps_span_so_the_arm_stays_readable():
    # A room-scale colored cloud must not shrink the arm into a corner: cloud
    # points past the work radius are excluded and the span is clamped so the arm
    # keeps a readable fraction of the view.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
scene.setLiveCameraPose(cameraPose);
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
// Wide near cloud (~2 m across, inside the work radius) plus a 3 m outlier that
// must be excluded from the fit entirely.
const wideXyz = new Float32Array([.4,1.0,0, .4,-1.0,0, .5,0,.3, 3.0,0,0]);
const wideRgb = new Uint8Array([255,0,0, 0,255,0, 0,0,255, 90,90,90]);
scene.setLiveColoredCloud(wideXyz, wideRgb, 4);
scene.flush();
const framing = scene.getState().framing;
console.log(JSON.stringify({{framing}}));
"""))

    framing = result["framing"]
    assert framing["source"] == "live"
    # Neither the 2 m-wide near cloud nor the 3 m outlier blew up the fit.
    assert 0.4 <= framing["span"] < 1.5


def test_session_bundle_keeps_live_colored_cloud_and_camera_without_refit():
    # Operator regression: activating a planning/grasp session while the live
    # view is up must (a) keep drawing the live colored cloud under the session
    # overlays, (b) keep accepting live cloud refreshes, and (c) never move the
    # operator's camera — orbit, framing, and the Z-up projection basis all
    # survive live -> session -> live for a same-frame bundle.  "Reset view"
    # remains the only recenter.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
scene.setLiveCameraPose(cameraPose);
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
// The operator has orbited/zoomed/panned away from the defaults.
scene.orbit.yaw = 1.23; scene.orbit.zoom = 2.5; scene.orbit.panX = 40;
const liveState = scene.getState();
const bundle = {{
  schema: 'z_manip.debug_bundle.v1',
  frames: {{ perception: 'camera', planning: 'piper_base_link' }},
  visualization: {{
    frame: 'piper_base_link', robot_overlay_allowed: true,
    scene_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.2,.1,0],[.3,-.1,.02]] }},
    target_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.32,.03,.21]] }},
    reference_axes: [{{ name: 'base', frame: 'piper_base_link', pose: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }}]
  }}
}};
operations.length = 0;
const accepted = scene.setBundle(bundle);
scene.flush();
const sessionState = scene.getState();
const drewCloudInSession = operations.some(op => op[0] === 'drawImage');
// Live cloud refreshes keep flowing while the session is displayed.
const refreshed = scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
const backState = scene.getState();
console.log(JSON.stringify({{
  liveState, accepted, sessionState, drewCloudInSession, refreshed, backState,
  diagnostics: scene.getDiagnostics().map(item => item.code)
}}));
"""))

    assert result["accepted"]["accepted"] is True
    assert result["accepted"]["frame"] == "piper_base_link"
    assert result["sessionState"]["live"] is False
    # (a) the live colored cloud persists and is composited in session mode.
    assert result["sessionState"]["counts"]["coloredCloudPoints"] == 4
    assert result["drewCloudInSession"] is True
    # (b) live refreshes are still accepted while session evidence is shown.
    assert result["refreshed"] == 4
    # (c) no viewpoint jump in either direction: orbit and framing are carried
    # verbatim (framing source stays "live" — no bundle_locked re-fit).
    assert result["sessionState"]["orbit"] == result["liveState"]["orbit"]
    assert result["sessionState"]["framing"] == result["liveState"]["framing"]
    assert result["backState"]["orbit"] == result["liveState"]["orbit"]
    assert result["backState"]["framing"] == result["liveState"]["framing"]
    assert result["backState"]["counts"]["coloredCloudPoints"] == 4


def test_live_colored_cloud_re_anchors_from_current_camera_pose_at_draw_time():
    # Operator regression (base approach): the environment cloud must sweep with
    # base motion, not stay glued to the arm.  The module stores the CAMERA-frame
    # cloud and composes the camera->base pose at DRAW time, so feeding two
    # successive camera poses re-projects the SAME untouched cloud arrays to
    # different pixels — with the locked virtual camera (framing) unchanged,
    # proving it is a draw-time re-anchor and not a view re-fit.  The anchor
    # keeps updating while SESSION evidence is displayed (continuity), where it
    # must not overwrite the bundle's capture-time camera frustum.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.setLiveCameraPose(cameraPose);
scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
scene.flush();
const framingA = scene.getState().framing;
const pixelA = scene._cloudU32.findIndex(v => v !== 0);
// The base advances: the hand-eye chain reports a new camera pose while the
// stored cloud arrays are untouched.
const poseB = cameraPose.map(row => row.slice());
poseB[0][3] += 0.12;
scene.setLiveCameraPose(poseB);
scene.flush();
const framingB = scene.getState().framing;
const pixelB = scene._cloudU32.findIndex(v => v !== 0);
// Session mode keeps re-anchoring the cloud but not the recorded frustum.
const bundle = {{
  schema: 'z_manip.debug_bundle.v1',
  frames: {{ perception: 'camera', planning: 'piper_base_link' }},
  visualization: {{
    frame: 'piper_base_link', robot_overlay_allowed: true,
    scene_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.2,.1,0],[.3,-.1,.02]] }},
    reference_axes: [
      {{ name: 'base', frame: 'piper_base_link', pose: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }},
      {{ name: 'camera', frame: 'piper_base_link', pose: cameraPose }}
    ]
  }}
}};
scene.setBundle(bundle);
const poseC = cameraPose.map(row => row.slice());
poseC[0][3] += 0.24;
const sessionPoseAccepted = scene.setLiveCameraPose(poseC);
scene.flush();
const pixelC = scene._cloudU32.findIndex(v => v !== 0);
const sessionState = scene.getState();
const frustumUntouched = scene.model.cameraPose === null
  || scene.model.cameraPose[0][3] === cameraPose[0][3];
console.log(JSON.stringify({{
  framingA, framingB, pixelA, pixelB, pixelC,
  sessionPoseAccepted, sessionState, frustumUntouched
}}));
"""))

    assert result["pixelA"] >= 0
    assert result["pixelB"] >= 0
    assert result["pixelC"] >= 0
    # Same cloud arrays, new pose, unchanged virtual camera => new pixels.
    assert result["framingA"] == result["framingB"]
    assert result["pixelA"] != result["pixelB"]
    # The anchor keeps moving while session evidence is displayed...
    assert result["sessionPoseAccepted"] is True
    assert result["pixelB"] != result["pixelC"]
    assert result["sessionState"]["counts"]["coloredCloudPoints"] == 4
    # ...without overwriting the bundle's capture-time camera frustum.
    assert result["frustumUntouched"] is True


def test_stale_feedback_tick_holds_skeleton_and_cloud_without_blanking():
    # Operator regression (grasp money shot): while a grasp executes, the arm
    # executor owns the CAN bus and the NUC passive-feedback service is suspended
    # BY DESIGN, so /piper/state joint feedback goes stale on EVERY grasp.  A
    # stale tick must FREEZE the last-known kinematic chain + colored cloud and
    # keep rendering everything — never blank the hero.  Even a downgraded overlay
    # gate (skeleton "unavailable") holds the frozen chain instead of tearing the
    # model down to an empty scene; only the hand-eye safety gate (cloudExpected)
    # may drop the colored cloud, and here it stays verified so the cloud holds.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
scene.setLiveCameraPose(cameraPose);
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
scene.flush();
const rendered = scene.getState();
operations.length = 0;
// Grasp begins: joint feedback stale.  The overlay gate is downgraded, yet the
// module must HOLD the frozen skeleton, colored cloud, anchor and framing.
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:false, cloudExpected:true}});
scene.flush();
const held = scene.getState();
const labels = operations.filter(op => op[0] === 'fillText').map(op => op[1]);
const drewCloud = operations.some(op => op[0] === 'drawImage');
console.log(JSON.stringify({{ rendered, held, labels, drewCloud }}));
"""))

    assert result["rendered"]["counts"]["actualLinks"] == 3
    assert result["rendered"]["counts"]["coloredCloudPoints"] == 4
    # Skeleton + cloud are HELD across the stale tick — nothing cleared.
    assert result["held"]["counts"]["actualLinks"] == 3
    assert result["held"]["counts"]["coloredCloudPoints"] == 4
    # Virtual camera held verbatim: no re-fit, no null framing, no empty state.
    assert result["held"]["framing"] == result["rendered"]["framing"]
    assert result["held"]["framing"]["source"] == "live"
    assert "Waiting for scene data" not in result["labels"]
    # Skeleton + cloud are still actually drawn during the hold.
    assert "base" in result["labels"]
    assert result["drewCloud"] is True


def test_empty_state_only_on_cold_start_never_after_first_geometry():
    # The full-screen empty state is allowed ONLY on a true cold start (no scene
    # data ever received this page load).  Once any geometry has been framed, a
    # later null-framing tick holds the last-known view instead of blanking.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
// Cold start: nothing has ever been received -> the empty state is allowed.
operations.length = 0;
scene.render();
const coldLabels = operations.filter(op => op[0] === 'fillText').map(op => op[1]);
// First geometry arrives; from now on the empty state must never return.
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:false}});
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.flush();
operations.length = 0;
scene.render();
const warmLabels = operations.filter(op => op[0] === 'fillText').map(op => op[1]);
console.log(JSON.stringify({{
  coldEmpty: coldLabels.includes('Waiting for scene data'),
  warmEmpty: warmLabels.includes('Waiting for scene data')
}}));
"""))

    assert result["coldEmpty"] is True
    assert result["warmEmpty"] is False


def test_mid_session_new_bundle_preserves_camera_and_colored_cloud():
    # Close-range handoff replans create FRESH bundles while the operator is
    # already viewing session evidence — typically at the exact moment the
    # hand-eye gate is DOWN (the transform vanishes with the suspended feedback
    # service mid-grasp).  A session -> session bundle swap must keep the
    # operator's virtual camera (orbit + framing) verbatim AND keep the live
    # colored cloud on stage: the held cloud was verified when it was fused, so
    # only a genuine frame mismatch may withhold it.
    result = _node(_live_harness(r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:true}});
scene.setLiveCameraPose(cameraPose);
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.setLiveColoredCloud(cloudXyz, cloudRgb, 4);
// The operator has orbited away from the defaults.
scene.orbit.yaw = 1.11; scene.orbit.zoom = 2.2; scene.orbit.panY = -25;
scene.flush();
const liveState = scene.getState();
const mkBundle = id => ({{
  schema: 'z_manip.debug_bundle.v1',
  frames: {{ perception: 'camera', planning: 'piper_base_link' }},
  visualization: {{
    frame: 'piper_base_link', robot_overlay_allowed: true,
    scene_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.2,.1,0],[.3 + id * .01,-.1,.02]] }},
    reference_axes: [{{ name: 'base', frame: 'piper_base_link', pose: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }}]
  }}
}});
// Grasp begins: feedback suspended, both caller gates drop BEFORE the session
// activates — the state every mid-grasp bundle is born into.
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:false, cloudExpected:false}});
const acceptedA = scene.setBundle(mkBundle(1));
scene.flush();
const sessionA = scene.getState();
operations.length = 0;
// Mid-session replan: a brand-new bundle arrives while session evidence is up.
const acceptedB = scene.setBundle(mkBundle(2));
scene.flush();
const sessionB = scene.getState();
const drewCloudAfterReplan = operations.some(op => op[0] === 'drawImage');
console.log(JSON.stringify({{
  liveState, acceptedA, sessionA, acceptedB, sessionB, drewCloudAfterReplan
}}));
"""))

    assert result["acceptedA"]["accepted"] is True
    assert result["acceptedB"]["accepted"] is True
    # Virtual camera bit-identical across live -> session -> new mid-session
    # bundle: orbit and framing never re-fit, never snap.
    assert result["sessionA"]["orbit"] == result["liveState"]["orbit"]
    assert result["sessionB"]["orbit"] == result["liveState"]["orbit"]
    assert result["sessionA"]["framing"] == result["liveState"]["framing"]
    assert result["sessionB"]["framing"] == result["liveState"]["framing"]
    # The colored cloud survives both session entries despite the closed gate,
    # and is actually composited after the replan.
    assert result["sessionA"]["counts"]["coloredCloudPoints"] == 4
    assert result["sessionB"]["counts"]["coloredCloudPoints"] == 4
    assert result["drewCloudAfterReplan"] is True


def test_session_view_orientation_is_bit_identical_to_live_view():
    # One canonical orientation, ever: the session/bundle view renders in the
    # SAME world orientation as the live view.  Projecting the same fixed world
    # point (the camera reference 0.4 m straight above the base) in live mode
    # and then in session mode must land on IDENTICAL raster positions — same
    # orbit, same framing, same Z-up basis.  A cold-start session (page loaded
    # mid-task, never live) must use the same canonical basis and default orbit.
    result = _node(_live_harness(r"""
const upPose = [[1,0,0,0],[0,1,0,0],[0,0,1,.4],[0,0,0,1]];
const bundle = {{
  schema: 'z_manip.debug_bundle.v1',
  frames: {{ perception: 'camera', planning: 'piper_base_link' }},
  visualization: {{
    frame: 'piper_base_link', robot_overlay_allowed: true,
    scene_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.2,.1,0],[.3,-.1,.02]] }},
    reference_axes: [
      {{ name: 'base', frame: 'piper_base_link', pose: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }},
      {{ name: 'camera', frame: 'piper_base_link', pose: upPose }}
    ]
  }}
}};
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:false}});
scene.setLiveCameraPose(upPose);
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.flush();
const findLabel = (ops, text) => ops.filter(op => op[0] === 'fillText').find(op => op[1] === text);
const liveBase = findLabel(operations, 'base');
const liveCam = findLabel(operations, 'camera pose');
operations.length = 0;
scene.setBundle(bundle);
scene.flush();
const sesBase = findLabel(operations, 'base');
const sesCam = findLabel(operations, 'camera pose');
// Cold start straight into session evidence (operator reloads mid-task).
operations.length = 0;
const scene2 = window.ZManipScene.create(new Canvas(), {{autoResize:false,interactive:false,reducedMotion:true}});
scene2.setBundle(bundle);
scene2.flush();
const coldBase = findLabel(operations, 'base');
const coldCam = findLabel(operations, 'camera pose');
console.log(JSON.stringify({{
  liveBase, liveCam, sesBase, sesCam, coldBase, coldCam,
  liveOrbit: scene.getState().orbit, coldOrbit: scene2.getState().orbit
}}));
"""))

    # Live -> session in one scene: identical raster positions for the same
    # world points (fillText payload is [text, x, y]).
    assert result["sesBase"][2:] == result["liveBase"][2:]
    assert result["sesCam"][2:] == result["liveCam"][2:]
    # Cold-start session uses the SAME canonical Z-up basis and default orbit as
    # live: the +z camera renders straight above the base, horizontally aligned.
    assert result["coldOrbit"] == result["liveOrbit"]
    assert result["coldCam"][3] < result["coldBase"][3] - 20
    assert abs(result["coldCam"][2] - result["coldBase"][2]) < 6


_COLD_LOAD_BUNDLE_JS = r"""
const bundle = {{
  schema: 'z_manip.debug_bundle.v1',
  frames: {{ perception: 'camera', planning: 'piper_base_link' }},
  visualization: {{
    frame: 'piper_base_link', robot_overlay_allowed: true,
    scene_cloud: {{ frame: 'piper_base_link', points_xyz_m: [[.2,.1,0],[.3,-.1,.02],[.25,0,.1]] }},
    reference_axes: [{{ name: 'base', frame: 'piper_base_link', pose: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }}]
  }}
}};
const emptyLabel = ops => ops.filter(op => op[0] === 'fillText').some(op => op[1] === 'Waiting for scene data');
const baseLabel = ops => ops.filter(op => op[0] === 'fillText').some(op => op[1] === 'base');
"""


def test_session_first_cold_load_frames_bundle_then_live():
    # Operator regression (mid-task page reload): the FIRST data the scene sees
    # is the session bundle — no live tick has arrived yet.  setBundle from the
    # never-framed cold state must produce the first framing and draw, and a
    # live entry afterwards must keep that framing (no empty state at any step).
    result = _node(_live_harness(_COLD_LOAD_BUNDLE_JS + r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
operations.length = 0;
const accepted = scene.setBundle(bundle);
scene.flush();
const framedFromBundle = scene.getState().framing;
const bundleEmpty = emptyLabel(operations);
const bundleDrewBase = baseLabel(operations);
operations.length = 0;
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:false}});
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.flush();
const framedLive = scene.getState().framing;
const liveEmpty = emptyLabel(operations);
const liveDrewBase = baseLabel(operations);
console.log(JSON.stringify({{
  accepted, framedFromBundle, bundleEmpty, bundleDrewBase, framedLive, liveEmpty, liveDrewBase
}}));
"""))

    assert result["accepted"]["accepted"] is True
    assert result["framedFromBundle"] is not None
    assert result["framedFromBundle"]["source"] == "bundle_locked"
    assert result["bundleEmpty"] is False
    assert result["bundleDrewBase"] is True
    # The bundle framing carries into live mode verbatim — still no empty state.
    assert result["framedLive"] == result["framedFromBundle"]
    assert result["liveEmpty"] is False
    assert result["liveDrewBase"] is True


def test_session_first_cold_load_frames_live_then_bundle():
    # The mirrored arrival order, replaying the REAL boot sequence: drawScene
    # first enters live mode with NO runtime data (gates closed, robot/pose
    # cleared to null), which renders the cold empty state — the only moment it
    # is allowed.  The session bundle then arrives and MUST frame and draw; a
    # later live tick keeps the scene framed.
    result = _node(_live_harness(_COLD_LOAD_BUNDLE_JS + r"""
const canvas = new Canvas();
const scene = window.ZManipScene.create(canvas, {{autoResize:false,interactive:false,reducedMotion:true}});
// Boot: applyLiveGeometry with state.runtime === null.
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:false, cloudExpected:false}});
scene.setLiveRobot(null);
scene.setLiveCameraPose(null);
operations.length = 0;
scene.render();
const coldEmpty = emptyLabel(operations);
// The session bundle arrives (ensureSessionScene hands it over on mode flip).
operations.length = 0;
const accepted = scene.setBundle(bundle);
scene.flush();
const framedFromBundle = scene.getState().framing;
const bundleEmpty = emptyLabel(operations);
const bundleDrewBase = baseLabel(operations);
// A later live tick (feedback returns) keeps the scene framed.
operations.length = 0;
scene.enterLiveMode({{frame:'piper_base_link', overlayAllowed:true, cloudExpected:false}});
scene.setLiveRobot({{frame:'piper_base_link', links_xyz_m: links}});
scene.flush();
const liveEmpty = emptyLabel(operations);
console.log(JSON.stringify({{
  coldEmpty, accepted, framedFromBundle, bundleEmpty, bundleDrewBase, liveEmpty
}}));
"""))

    assert result["coldEmpty"] is True
    assert result["accepted"]["accepted"] is True
    assert result["framedFromBundle"] is not None
    assert result["bundleEmpty"] is False
    assert result["bundleDrewBase"] is True
    assert result["liveEmpty"] is False


def test_dashboard_hands_bundle_to_scene_when_session_mode_engages_late():
    # Orchestration half of the same regression: on a mid-task reload the boot
    # order (loadBundle -> refreshSessionStatus -> refreshRuntime) resolves the
    # geometry mode BEFORE taskContextActive flips true, so renderBundle skips
    # its setBundle and no remaining path ever fed the renderer — the hero sat
    # in the cold empty state forever while every other panel streamed.  The
    # dashboard must hand the bundle over whenever session mode is effective but
    # the module is not yet displaying session evidence, and must redraw on a
    # task-context flip.
    html = (ROOT / "web/debug_dashboard/index.html").read_text(encoding="utf-8")
    assert "function ensureSessionScene" in html
    # Definition plus BOTH per-tick session paths (syncGeometryView + drawScene).
    assert html.count("ensureSessionScene(") >= 3
    assert "if (contextWasActive !== contextActive) drawScene();" in html


def test_javascript_syntax_is_valid():
    subprocess.run(["node", "--check", str(SCRIPT)], check=True)
