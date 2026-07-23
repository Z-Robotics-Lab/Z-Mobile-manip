/* Z-Manip dependency-free Canvas2D runtime scene renderer.
 *
 * Browser API:
 *   const scene = window.ZManipScene.create(canvas, options);
 *   scene.setBundle(debugBundle);
 *   scene.setJointState({frame, links_xyz_m, joint_positions_rad});
 *   scene.setPlannedState({frame, links_xyz_m});
 *   scene.update({selectedCandidateId, collisionWitness, ...});
 *   scene.setSelection({candidateId, rejection});
 *   scene.resize(); scene.resetView(); scene.flush(); scene.destroy();
 *
 * All overlays are observation-only.  The module contains no network, ROS,
 * robot transport, command publication, model inference, or kinematics code.
 */
(function installZManipScene(root) {
  "use strict";

  const VERSION = "1.0.0";
  const DEFAULTS = Object.freeze({
    maxFps: 30,
    maxCloudPoints: 5000,
    cloudVoxelM: 0.005,
    showAllCandidates: false,
    autoResize: true,
    interactive: true,
    reducedMotion: null,
    background: "#070b0e",
    grid: "#1b2a32",
    scenePoint: "#68767d",
    targetPoint: "#49d9df",
    actualRobot: "#d8e2e6",
    plannedRobot: "#6f9dff",
    collision: "#f0b758",
    witness: "#ff6673",
  });
  const AXIS_COLORS = Object.freeze(["#ff6673", "#54d68c", "#659cff"]);
  const PATH_COLORS = Object.freeze({
    transit: "#659cff",
    approach: "#f0b758",
    lift: "#54d68c",
  });
  const IDENTITY = Object.freeze([
    Object.freeze([1, 0, 0, 0]),
    Object.freeze([0, 1, 0, 0]),
    Object.freeze([0, 0, 1, 0]),
    Object.freeze([0, 0, 0, 1]),
  ]);

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function point(value) {
    return Array.isArray(value)
      && value.length >= 3
      && value.slice(0, 3).every(finite)
      ? [value[0], value[1], value[2]]
      : null;
  }

  function pose(value) {
    if (!Array.isArray(value) || value.length !== 4) return null;
    const rows = value.map(row => (
      Array.isArray(row) && row.length === 4 && row.every(finite)
        ? row.slice()
        : null
    ));
    if (rows.some(row => row === null)) return null;
    if (
      Math.abs(rows[3][0]) > 1e-6
      || Math.abs(rows[3][1]) > 1e-6
      || Math.abs(rows[3][2]) > 1e-6
      || Math.abs(rows[3][3] - 1) > 1e-6
    ) return null;
    return rows;
  }

  function origin(matrix) {
    return matrix ? [matrix[0][3], matrix[1][3], matrix[2][3]] : null;
  }

  function transformPoint(matrix, value) {
    return [
      matrix[0][0] * value[0] + matrix[0][1] * value[1] + matrix[0][2] * value[2] + matrix[0][3],
      matrix[1][0] * value[0] + matrix[1][1] * value[1] + matrix[1][2] * value[2] + matrix[1][3],
      matrix[2][0] * value[0] + matrix[2][1] * value[1] + matrix[2][2] * value[2] + matrix[2][3],
    ];
  }

  function normalizeFrame(value) {
    return typeof value === "string" && value.trim() ? value.trim() : null;
  }

  function validateFrame(displayFrame, inputFrame) {
    const display = normalizeFrame(displayFrame);
    const input = normalizeFrame(inputFrame);
    return display !== null && input !== null && display === input;
  }

  function clockNow() {
    if (root.performance && typeof root.performance.now === "function") {
      return root.performance.now();
    }
    return Date.now();
  }

  function cloneOptions(options) {
    return Object.assign({}, DEFAULTS, options || {});
  }

  function reducedMotionPreference(options) {
    if (typeof options.reducedMotion === "boolean") return options.reducedMotion;
    return Boolean(
      root.matchMedia
      && root.matchMedia("(prefers-reduced-motion: reduce)").matches,
    );
  }

  class RuntimeScene {
    constructor(canvas, options) {
      if (!canvas || typeof canvas.getContext !== "function") {
        throw new TypeError("ZManipScene.create requires a canvas element");
      }
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas2D is unavailable");
      this.canvas = canvas;
      this.context = context;
      this.options = cloneOptions(options);
      this.reducedMotion = reducedMotionPreference(this.options);
      this.maxFps = Math.max(1, Math.min(60, Number(this.options.maxFps) || 30));
      this.displayFrame = null;
      this.bundle = null;
      this.model = this._emptyModel();
      this.diagnostics = [];
      this._diagnosticKeys = new Set();
      this._destroyed = false;
      this._pendingHandle = null;
      this._pendingKind = null;
      this._lastRenderMs = -Infinity;
      this._renderCount = 0;
      this._selection = { candidateId: null, rejection: null };
      this._framing = null;
      this.orbit = { yaw: -0.72, pitch: -0.46, zoom: 1, panX: 0, panY: 0 };
      this._drag = null;
      this._listeners = [];
      this._observer = null;
      this._viewport = { width: 1, height: 1, ratio: 1 };
      this._filterStats = this._emptyFilterStats();
      // Live view (subscribe-only observer feed, no recorded bundle).  The
      // colored cloud is rasterized into an offscreen buffer and composited
      // under the vector overlays so 15-20k points never stall the page.
      this.live = false;
      this._liveFramingLocked = false;
      this._cloudCanvas = null;
      this._cloudCtx = null;
      this._cloudImage = null;
      this._cloudU32 = null;
      this._cloudZ = null;
      if (this.options.interactive) this._bindInteractions();
      if (this.options.autoResize) this._bindResize();
      this.resize(false);
      this.render();
    }

    _emptyModel() {
      return {
        scene: [],
        target: [],
        candidates: [],
        references: [],
        actualRobot: null,
        plannedRobot: null,
        trajectory: {},
        graspArrow: null,
        collision: null,
        basePose: null,
        cameraPose: null,
        coloredCloud: null,
        overlayAllowed: false,
      };
    }

    _emptyFilterStats() {
      return {
        displayOnly: true,
        cloudVoxelM: Math.max(0.0001, Number(this.options.cloudVoxelM) || 0.005),
        cloudUpdatesSuppressed: 0,
        clouds: {},
      };
    }

    _diagnose(code, message, path) {
      const key = `${code}:${path || ""}:${message}`;
      if (this._diagnosticKeys.has(key)) return;
      this._diagnosticKeys.add(key);
      const item = { code, message, path: path || null };
      this.diagnostics.push(item);
      if (this.diagnostics.length > 100) this.diagnostics.shift();
      if (typeof this.options.onDiagnostic === "function") {
        try { this.options.onDiagnostic(item); } catch (_) { /* fail-soft callback */ }
      }
    }

    _acceptFrame(inputFrame, path, inherited) {
      const frame = normalizeFrame(inputFrame) || (inherited ? this.displayFrame : null);
      if (!frame) {
        this._diagnose("MISSING_FRAME", "overlay has no explicit frame", path);
        return false;
      }
      if (!validateFrame(this.displayFrame, frame)) {
        this._diagnose(
          "FRAME_MISMATCH",
          `expected ${this.displayFrame || "<unset>"}, received ${frame}`,
          path,
        );
        return false;
      }
      return true;
    }

    _samplePoints(values, path) {
      if (!Array.isArray(values)) {
        this._diagnose("INVALID_POINTS", "point array is unavailable", path);
        return [];
      }
      const voxel = this._filterStats.cloudVoxelM;
      const voxels = new Map();
      let validInputPoints = 0;
      for (const value of values) {
        const parsed = point(value);
        if (!parsed) continue;
        validInputPoints += 1;
        const key = parsed.map(coordinate => Math.round(coordinate / voxel));
        voxels.set(key.join(","), key);
      }
      const quantized = [...voxels.values()].sort((left, right) => (
        left[0] - right[0] || left[1] - right[1] || left[2] - right[2]
      )).map(key => key.map(component => component * voxel));
      const maximum = Math.max(1, Number(this.options.maxCloudPoints) || 5000);
      let valid = quantized;
      if (quantized.length > maximum) {
        if (maximum === 1) {
          valid = [quantized[Math.floor(quantized.length / 2)]];
        } else {
          valid = [];
          for (let index = 0; index < maximum; index += 1) {
            const source = Math.round(index * (quantized.length - 1) / (maximum - 1));
            valid.push(quantized[source]);
          }
        }
      }
      this._filterStats.clouds[path] = {
        inputPoints: values.length,
        validInputPoints,
        quantizedVoxels: quantized.length,
        displayPoints: valid.length,
      };
      if (values.length && !valid.length) {
        this._diagnose("INVALID_POINTS", "point array has no finite XYZ rows", path);
      }
      return valid;
    }

    _samePoints(first, second) {
      if (!Array.isArray(first) || !Array.isArray(second) || first.length !== second.length) {
        return false;
      }
      return first.every((value, index) => (
        value.length === second[index].length
        && value.every((coordinate, axis) => coordinate === second[index][axis])
      ));
    }

    _cloud(value, path) {
      if (!value || typeof value !== "object") return [];
      if (!this._acceptFrame(value.frame, path, false)) return [];
      return this._samplePoints(value.points_xyz_m, `${path}.points_xyz_m`);
    }

    _robot(value, path, requireOverlayGate) {
      if (!value || typeof value !== "object") return null;
      if (requireOverlayGate && !this.model.overlayAllowed) {
        this._diagnose("OVERLAY_LOCKED", "robot overlay is disabled by calibration gate", path);
        return null;
      }
      if (!this._acceptFrame(value.frame, path, false)) return null;
      const links = [];
      if (Array.isArray(value.links_xyz_m)) {
        for (const candidate of value.links_xyz_m) {
          if (!Array.isArray(candidate) || candidate.length !== 2) continue;
          const start = point(candidate[0]);
          const end = point(candidate[1]);
          if (start && end) links.push([start, end]);
        }
      }
      if (!links.length) {
        this._diagnose("INVALID_ROBOT_OVERLAY", "robot overlay has no finite link segments", path);
        return null;
      }
      const joints = [];
      const suppliedJoints = Array.isArray(value.joints_xyz_m) ? value.joints_xyz_m : [];
      for (const candidate of suppliedJoints) {
        const parsed = point(candidate);
        if (parsed) joints.push(parsed);
      }
      if (!joints.length) {
        joints.push(links[0][0]);
        for (const link of links) joints.push(link[1]);
      }
      return {
        frame: this.displayFrame,
        links,
        joints,
        jointPositions: Array.isArray(value.joint_positions_rad)
          ? value.joint_positions_rad.filter(finite)
          : [],
        gripper: value.gripper && typeof value.gripper === "object" ? value.gripper : null,
        poseSource: value.pose_source || null,
      };
    }

    _candidateAxes(values) {
      if (!Array.isArray(values)) return [];
      const output = [];
      for (let index = 0; index < values.length; index += 1) {
        const item = values[index];
        if (!item || typeof item !== "object") continue;
        if (!this._acceptFrame(item.frame, `visualization.candidate_axes[${index}]`, false)) continue;
        const matrix = pose(item.pose);
        if (!matrix) {
          this._diagnose("INVALID_POSE", "candidate pose is not a finite transform", `visualization.candidate_axes[${index}].pose`);
          continue;
        }
        output.push({
          candidateId: Number.isInteger(item.candidate_id) ? item.candidate_id : index,
          rank: Number.isInteger(item.rank) ? item.rank : null,
          status: typeof item.status === "string" ? item.status : "unknown",
          pose: matrix,
        });
      }
      return output;
    }

    _references(values) {
      if (!Array.isArray(values)) return [];
      const output = [];
      for (let index = 0; index < values.length; index += 1) {
        const item = values[index];
        if (!item || typeof item !== "object") continue;
        if (!this._acceptFrame(item.frame, `visualization.reference_axes[${index}]`, false)) continue;
        const matrix = pose(item.pose);
        if (!matrix) continue;
        output.push({ name: String(item.name || `reference-${index}`), pose: matrix });
      }
      return output;
    }

    _trajectory(value, path, inheritedFrame) {
      if (!value || typeof value !== "object") return {};
      const frame = value.frame;
      if (!this._acceptFrame(frame, path, inheritedFrame)) return {};
      const source = value.segments && typeof value.segments === "object" ? value.segments : value;
      const output = {};
      for (const name of Object.keys(PATH_COLORS)) {
        const values = Array.isArray(source[name])
          ? source[name]
          : source[name] && Array.isArray(source[name].points_xyz_m)
            ? source[name].points_xyz_m
            : [];
        const points = values.map(point).filter(Boolean);
        if (points.length) output[name] = points;
      }
      return output;
    }

    _graspArrow(selectedPlan, planningFrame) {
      if (!selectedPlan || typeof selectedPlan !== "object") return null;
      if (!this.model.overlayAllowed) return null;
      if (!this._acceptFrame(planningFrame, "selected_plan", false)) return null;
      const pregrasp = pose(selectedPlan.pregrasp_pose_base);
      const grasp = pose(selectedPlan.grasp_pose_base);
      if (!pregrasp || !grasp) return null;
      return { pregrasp, grasp };
    }

    _collision(value, path) {
      if (!value || typeof value !== "object") return null;
      if (!this._acceptFrame(value.frame, path, false)) return null;
      const capsuleStart = point(value.capsule_start_base || value.capsule_start);
      const capsuleEnd = point(value.capsule_end_base || value.capsule_end);
      const scenePoint = point(value.witness_scene_point_base || value.scene_point);
      const capsulePoint = point(value.witness_capsule_point_base || value.capsule_point);
      if (!capsuleStart || !capsuleEnd || !scenePoint || !capsulePoint) {
        this._diagnose(
          "INVALID_COLLISION_WITNESS",
          "collision witness needs capsule endpoints plus both nearest points",
          path,
        );
        return null;
      }
      return {
        capsuleStart,
        capsuleEnd,
        scenePoint,
        capsulePoint,
        radiusM: finite(value.capsule_radius_m)
          ? Math.max(0, value.capsule_radius_m)
          : finite(value.radius_m)
            ? Math.max(0, value.radius_m)
          : finite(value.threshold_m)
            ? Math.max(0, value.threshold_m)
            : 0.04,
        distanceM: finite(value.distance_m) ? value.distance_m : null,
        thresholdM: finite(value.threshold_m) ? value.threshold_m : null,
        kind: value.kind || "collision",
      };
    }

    _defaultWitness(bundle, candidateId) {
      if (!Number.isInteger(candidateId)) return null;
      const rejections = bundle && bundle.planning && Array.isArray(bundle.planning.rejections)
        ? bundle.planning.rejections
        : [];
      const matching = rejections.find(item => (
        item
        && item.collision_witness
        && item.candidate_index === candidateId
      ));
      return matching ? matching.collision_witness : null;
    }

    setBundle(bundle) {
      if (this._destroyed) return { accepted: false, diagnostics: this.getDiagnostics() };
      this.diagnostics = [];
      this._diagnosticKeys.clear();
      this.bundle = bundle && typeof bundle === "object" ? bundle : null;
      this.model = this._emptyModel();
      this._filterStats = this._emptyFilterStats();
      this._framing = null;
      const visualization = this.bundle && this.bundle.visualization;
      this.displayFrame = normalizeFrame(visualization && visualization.frame);
      if (!this.displayFrame) {
        this._diagnose("MISSING_DISPLAY_FRAME", "visualization.frame is required", "visualization.frame");
        this.render();
        return { accepted: false, frame: null, diagnostics: this.getDiagnostics() };
      }
      this.model.overlayAllowed = visualization.robot_overlay_allowed === true;
      this.model.scene = this._cloud(visualization.scene_cloud, "visualization.scene_cloud");
      this.model.target = this._cloud(visualization.target_cloud, "visualization.target_cloud");
      this.model.candidates = this._candidateAxes(visualization.candidate_axes);
      this.model.references = this._references(visualization.reference_axes);
      this.model.basePose = (
        this.model.references.find(item => item.name === "base") || {}
      ).pose || (
        normalizeFrame(this.bundle.frames && this.bundle.frames.planning) === this.displayFrame
          ? IDENTITY.map(row => row.slice())
          : null
      );
      this.model.cameraPose = (
        this.model.references.find(item => item.name === "camera") || {}
      ).pose || null;
      if (!this.model.basePose) {
        this._diagnose("BASE_FRAME_UNAVAILABLE", "floor grid and base axes are hidden", "visualization.reference_axes");
      }
      this.model.actualRobot = this._robot(
        visualization.robot_overlay,
        "visualization.robot_overlay",
        true,
      );
      this.model.plannedRobot = this._robot(
        visualization.planned_robot_overlay,
        "visualization.planned_robot_overlay",
        true,
      );
      this.model.trajectory = this._trajectory(
        visualization.trajectory_xyz_m,
        "visualization.trajectory_xyz_m",
        true,
      );
      const selectedCandidate = this.bundle.selected_plan
        && Number.isInteger(this.bundle.selected_plan.candidate_id)
        ? this.bundle.selected_plan.candidate_id
        : (this.model.candidates.find(item => item.status === "selected") || {}).candidateId;
      this._selection.candidateId = Number.isInteger(selectedCandidate) ? selectedCandidate : null;
      this._selection.rejection = null;
      this.model.graspArrow = this._graspArrow(
        this.bundle.selected_plan,
        this.bundle.frames && this.bundle.frames.planning,
      );
      this.model.collision = this._collision(
        this._defaultWitness(this.bundle, this._selection.candidateId),
        "planning.rejections.collision_witness",
      );
      // Lock auto-framing to the recorded planning bundle. Live depth clouds
      // contain edge noise and outliers; fitting the virtual camera to every
      // runtime update makes a stationary robot appear to shake.
      this._framing = this._fitLockedFraming();
      this.render();
      return {
        accepted: true,
        frame: this.displayFrame,
        overlayAllowed: this.model.overlayAllowed,
        diagnostics: this.getDiagnostics(),
      };
    }

    update(values) {
      if (this._destroyed || !values || typeof values !== "object") return false;
      let changed = false;
      if (Object.prototype.hasOwnProperty.call(values, "actualRobot")) {
        const robot = this._robot(values.actualRobot, "update.actualRobot", true);
        if (robot) { this.model.actualRobot = robot; changed = true; }
      }
      if (Object.prototype.hasOwnProperty.call(values, "plannedRobot")) {
        const robot = this._robot(values.plannedRobot, "update.plannedRobot", true);
        if (robot) { this.model.plannedRobot = robot; changed = true; }
      }
      if (Object.prototype.hasOwnProperty.call(values, "jointState")) {
        const robot = this._robot(values.jointState, "update.jointState", true);
        if (robot) { this.model.actualRobot = robot; changed = true; }
      }
      if (Object.prototype.hasOwnProperty.call(values, "sceneCloud")) {
        const cloud = this._cloud(values.sceneCloud, "update.sceneCloud");
        if (cloud.length && this._samePoints(this.model.scene, cloud)) {
          this._filterStats.cloudUpdatesSuppressed += 1;
        } else if (cloud.length) { this.model.scene = cloud; changed = true; }
      }
      if (Object.prototype.hasOwnProperty.call(values, "targetCloud")) {
        const cloud = this._cloud(values.targetCloud, "update.targetCloud");
        if (cloud.length && this._samePoints(this.model.target, cloud)) {
          this._filterStats.cloudUpdatesSuppressed += 1;
        } else if (cloud.length) { this.model.target = cloud; changed = true; }
      }
      if (Object.prototype.hasOwnProperty.call(values, "trajectory")) {
        const trajectory = this._trajectory(values.trajectory, "update.trajectory", false);
        if (Object.keys(trajectory).length) { this.model.trajectory = trajectory; changed = true; }
      }
      if (Object.prototype.hasOwnProperty.call(values, "collisionWitness")) {
        const collision = this._collision(values.collisionWitness, "update.collisionWitness");
        this.model.collision = collision;
        changed = true;
      }
      if (Number.isInteger(values.selectedCandidateId)) {
        this._selection.candidateId = values.selectedCandidateId;
        changed = true;
      }
      if (changed) this._scheduleRender();
      return changed;
    }

    setJointState(value) {
      return this.update({ jointState: value });
    }

    setPlannedState(value) {
      return this.update({ plannedRobot: value });
    }

    setSelection(value) {
      const selection = value && typeof value === "object" ? value : {};
      if (Object.prototype.hasOwnProperty.call(selection, "candidateId")) {
        this._selection.candidateId = Number.isInteger(selection.candidateId)
          ? selection.candidateId
          : null;
      }
      this._selection.rejection = selection.rejection || null;
      const witness = selection.rejection && selection.rejection.collision_witness
        ? selection.rejection.collision_witness
        : this._defaultWitness(this.bundle, this._selection.candidateId);
      this.model.collision = this._collision(witness, "selection.collision_witness");
      this._scheduleRender();
      return this.getState();
    }

    // --- Live view -------------------------------------------------------
    // The live view renders the same base-frame geometry language as a recorded
    // session (floor, base axes, wrist-camera frustum, arm skeleton) but is fed
    // by the subscribe-only runtime observer instead of an immutable bundle, and
    // a colored point cloud replaces the sparse monochrome scene cloud.  Every
    // input stays frame-gated: the caller supplies base-frame geometry only when
    // the measured hand-eye transform is verified, otherwise it supplies a
    // camera-frame cloud with the base overlays left locked.
    enterLiveMode(options) {
      if (this._destroyed) return { accepted: false, frame: null, live: false };
      const opts = options && typeof options === "object" ? options : {};
      const frame = normalizeFrame(opts.frame);
      if (!frame) {
        this._diagnose("MISSING_DISPLAY_FRAME", "live view requires an explicit display frame", "live.frame");
        return { accepted: false, frame: null, live: false };
      }
      const overlayAllowed = opts.overlayAllowed === true;
      if (this.live && this.displayFrame === frame && this.model.overlayAllowed === overlayAllowed) {
        return { accepted: true, frame, overlayAllowed, live: true };
      }
      this.live = true;
      this.bundle = null;
      this.diagnostics = [];
      this._diagnosticKeys.clear();
      this.displayFrame = frame;
      this.model = this._emptyModel();
      this.model.overlayAllowed = overlayAllowed;
      this._filterStats = this._emptyFilterStats();
      this._selection = { candidateId: null, rejection: null };
      // The display frame is the base frame origin, so the floor grid and base
      // axes anchor at identity.
      this.model.basePose = IDENTITY.map(row => row.slice());
      this._framing = null;
      this._liveFramingLocked = false;
      this._scheduleRender();
      return { accepted: true, frame, overlayAllowed, live: true };
    }

    setLiveRobot(value) {
      if (this._destroyed || !this.live) return false;
      if (value === null) {
        const changed = this.model.actualRobot !== null;
        this.model.actualRobot = null;
        if (changed) this._scheduleRender();
        return false;
      }
      const robot = this._robot(value, "live.robot", true);
      if (!robot) return false;
      this.model.actualRobot = robot;
      this._ensureLiveFraming();
      this._scheduleRender();
      return true;
    }

    setLiveCameraPose(value) {
      if (this._destroyed || !this.live) return false;
      const matrix = value === null ? null : pose(value);
      if (value !== null && !matrix) {
        this._diagnose("INVALID_POSE", "live camera pose is not a finite transform", "live.cameraPose");
        return false;
      }
      this.model.cameraPose = matrix;
      this._ensureLiveFraming();
      this._scheduleRender();
      return Boolean(matrix);
    }

    setLiveColoredCloud(xyz, rgb, count) {
      if (this._destroyed || !this.live) return 0;
      const total = Number.isInteger(count) && count > 0 ? count : 0;
      const usable = total > 0
        && xyz && typeof xyz.length === "number" && xyz.length >= total * 3
        && rgb && typeof rgb.length === "number" && rgb.length >= total * 3;
      if (!usable) {
        const changed = this.model.coloredCloud !== null;
        this.model.coloredCloud = null;
        if (changed) this._scheduleRender();
        return 0;
      }
      this.model.coloredCloud = { xyz, rgb, count: total };
      this._ensureLiveFraming();
      this._scheduleRender();
      return total;
    }

    _ensureLiveFraming() {
      if (!this.live || this._liveFramingLocked) return;
      const framing = this._fitLiveFraming();
      if (!framing) return;
      this._framing = framing;
      // Lock the virtual camera once the full live geometry is present so later
      // joint motion and cloud refreshes cannot shake a stationary view.
      if (this.model.actualRobot && this.model.cameraPose && this.model.coloredCloud) {
        this._liveFramingLocked = true;
      }
    }

    _fitLiveFraming() {
      const focus = [];
      if (this.model.actualRobot) {
        for (const link of this.model.actualRobot.links) focus.push(link[0], link[1]);
      }
      if (this.model.basePose) focus.push(origin(this.model.basePose));
      if (this.model.cameraPose) focus.push(origin(this.model.cameraPose));
      const cloud = this.model.coloredCloud;
      if (cloud && cloud.count) {
        const step = Math.max(1, Math.floor(cloud.count / 1500));
        const sample = [];
        for (let index = 0; index < cloud.count; index += step) {
          const base = index * 3;
          const value = [cloud.xyz[base], cloud.xyz[base + 1], cloud.xyz[base + 2]];
          if (value.every(finite)) sample.push(value);
        }
        for (const value of this._robustCloudPoints(sample)) focus.push(value);
      }
      const values = focus.filter(Boolean);
      if (values.length < 2) return null;
      const low = [Infinity, Infinity, Infinity];
      const high = [-Infinity, -Infinity, -Infinity];
      for (const value of values) {
        for (let axis = 0; axis < 3; axis += 1) {
          low[axis] = Math.min(low[axis], value[axis]);
          high[axis] = Math.max(high[axis], value[axis]);
        }
      }
      const center = low.map((value, axis) => (value + high[axis]) * 0.5);
      const span = Math.max(...low.map((value, axis) => high[axis] - value), 0.28);
      return { center, span, source: "live" };
    }

    _ensureCloudBuffer() {
      const doc = (typeof root.document !== "undefined" && root.document
        && typeof root.document.createElement === "function") ? root.document : null;
      if (!doc) return null;
      if (!this._cloudCanvas) {
        this._cloudCanvas = doc.createElement("canvas");
        this._cloudCtx = this._cloudCanvas && typeof this._cloudCanvas.getContext === "function"
          ? this._cloudCanvas.getContext("2d")
          : null;
      }
      if (!this._cloudCtx || typeof this._cloudCtx.createImageData !== "function") return null;
      const pixelWidth = this.canvas.width;
      const pixelHeight = this.canvas.height;
      if (!(pixelWidth >= 1) || !(pixelHeight >= 1)) return null;
      if (
        this._cloudCanvas.width !== pixelWidth
        || this._cloudCanvas.height !== pixelHeight
        || !this._cloudImage
      ) {
        this._cloudCanvas.width = pixelWidth;
        this._cloudCanvas.height = pixelHeight;
        this._cloudImage = this._cloudCtx.createImageData(pixelWidth, pixelHeight);
        if (!this._cloudImage || !this._cloudImage.data || !this._cloudImage.data.buffer) {
          this._cloudImage = null;
          return null;
        }
        this._cloudU32 = new Uint32Array(this._cloudImage.data.buffer);
        this._cloudZ = new Float32Array(pixelWidth * pixelHeight);
      }
      return this._cloudCanvas;
    }

    _drawColoredCloud(view) {
      const cloud = this.model.coloredCloud;
      if (!cloud || !cloud.count || !view) return;
      const surface = this._ensureCloudBuffer();
      if (!surface) return;
      const pixelWidth = this._cloudCanvas.width;
      const pixelHeight = this._cloudCanvas.height;
      const u32 = this._cloudU32;
      const zbuf = this._cloudZ;
      u32.fill(0);
      zbuf.fill(Infinity);
      const ratio = this._viewport.ratio;
      const center = view.center;
      const scale = Math.min(this._viewport.width, this._viewport.height) * 0.80 / view.span * this.orbit.zoom;
      const cyaw = Math.cos(this.orbit.yaw);
      const syaw = Math.sin(this.orbit.yaw);
      const cpit = Math.cos(this.orbit.pitch);
      const spit = Math.sin(this.orbit.pitch);
      const halfW = this._viewport.width * 0.5 + this.orbit.panX;
      const halfH = this._viewport.height * 0.5 + this.orbit.panY;
      const cx = center[0];
      const cy = center[1];
      const cz = center[2];
      const xyz = cloud.xyz;
      const rgb = cloud.rgb;
      const count = cloud.count;
      const size = Math.max(1, Math.round(ratio * 1.5));
      for (let index = 0; index < count; index += 1) {
        const base = index * 3;
        const wx = xyz[base] - cx;
        const wy = xyz[base + 1] - cy;
        const wz = xyz[base + 2] - cz;
        const x1 = cyaw * wx - syaw * wz;
        const z1 = syaw * wx + cyaw * wz;
        const y2 = cpit * wy - spit * z1;
        const z2 = spit * wy + cpit * z1;
        const sx = ((halfW + x1 * scale) * ratio) | 0;
        const sy = ((halfH - y2 * scale) * ratio) | 0;
        if (sx < 0 || sy < 0 || sx >= pixelWidth || sy >= pixelHeight) continue;
        // Little-endian RGBA packed as one uint32 (A=255,B,G,R).
        const pixel = (255 << 24) | (rgb[base + 2] << 16) | (rgb[base + 1] << 8) | rgb[base];
        for (let dy = 0; dy < size; dy += 1) {
          const py = sy + dy;
          if (py >= pixelHeight) break;
          const rowBase = py * pixelWidth;
          for (let dx = 0; dx < size; dx += 1) {
            const px = sx + dx;
            if (px >= pixelWidth) break;
            const idx = rowBase + px;
            if (z2 < zbuf[idx]) {
              zbuf[idx] = z2;
              u32[idx] = pixel;
            }
          }
        }
      }
      this._cloudCtx.putImageData(this._cloudImage, 0, 0);
      this.context.drawImage(this._cloudCanvas, 0, 0, this._viewport.width, this._viewport.height);
    }

    _scheduleRender() {
      if (this._destroyed) return;
      if (this.reducedMotion) {
        this.flush();
        return;
      }
      if (this._pendingHandle !== null) return;
      const interval = 1000 / this.maxFps;
      const delay = Math.max(0, interval - (clockNow() - this._lastRenderMs));
      if (delay <= 1 && typeof root.requestAnimationFrame === "function") {
        this._pendingKind = "raf";
        this._pendingHandle = root.requestAnimationFrame(() => {
          this._pendingHandle = null;
          this._pendingKind = null;
          this.render();
        });
      } else {
        this._pendingKind = "timeout";
        this._pendingHandle = root.setTimeout(() => {
          this._pendingHandle = null;
          this._pendingKind = null;
          this.render();
        }, delay);
      }
    }

    _cancelPending() {
      if (this._pendingHandle === null) return;
      if (this._pendingKind === "raf" && typeof root.cancelAnimationFrame === "function") {
        root.cancelAnimationFrame(this._pendingHandle);
      } else {
        root.clearTimeout(this._pendingHandle);
      }
      this._pendingHandle = null;
      this._pendingKind = null;
    }

    flush() {
      this._cancelPending();
      this.render();
      return this;
    }

    _bindInteractions() {
      const add = (target, event, handler, options) => {
        if (!target || typeof target.addEventListener !== "function") return;
        target.addEventListener(event, handler, options);
        this._listeners.push(() => target.removeEventListener(event, handler, options));
      };
      add(this.canvas, "pointerdown", event => {
        if (typeof this.canvas.setPointerCapture === "function") {
          this.canvas.setPointerCapture(event.pointerId);
        }
        this._drag = { x: event.clientX, y: event.clientY, button: event.button };
      });
      add(this.canvas, "pointermove", event => {
        if (!this._drag) return;
        const dx = event.clientX - this._drag.x;
        const dy = event.clientY - this._drag.y;
        this._drag.x = event.clientX;
        this._drag.y = event.clientY;
        if (event.shiftKey || this._drag.button === 1) {
          this.orbit.panX += dx;
          this.orbit.panY += dy;
        } else {
          this.orbit.yaw += dx * 0.008;
          this.orbit.pitch = Math.max(-1.45, Math.min(1.45, this.orbit.pitch + dy * 0.008));
        }
        this._scheduleRender();
      });
      const release = () => { this._drag = null; };
      add(this.canvas, "pointerup", release);
      add(this.canvas, "pointercancel", release);
      add(this.canvas, "wheel", event => {
        if (typeof event.preventDefault === "function") event.preventDefault();
        this.orbit.zoom = Math.max(
          0.2,
          Math.min(10, this.orbit.zoom * Math.exp(-event.deltaY * 0.001)),
        );
        this._scheduleRender();
      }, { passive: false });
    }

    _bindResize() {
      if (typeof root.ResizeObserver === "function") {
        this._observer = new root.ResizeObserver(() => this.resize());
        this._observer.observe(this.canvas);
      } else if (typeof root.addEventListener === "function") {
        const handler = () => this.resize();
        root.addEventListener("resize", handler);
        this._listeners.push(() => root.removeEventListener("resize", handler));
      }
    }

    resize(renderAfter) {
      if (this._destroyed) return this._viewport;
      const rectangle = typeof this.canvas.getBoundingClientRect === "function"
        ? this.canvas.getBoundingClientRect()
        : { width: this.canvas.clientWidth || this.canvas.width || 1, height: this.canvas.clientHeight || this.canvas.height || 1 };
      const width = Math.max(1, Math.round(rectangle.width || 1));
      const height = Math.max(1, Math.round(rectangle.height || 1));
      const ratio = Math.max(1, Math.min(Number(root.devicePixelRatio) || 1, 2));
      this._viewport = { width, height, ratio };
      const pixelWidth = Math.round(width * ratio);
      const pixelHeight = Math.round(height * ratio);
      if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
      if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
      if (renderAfter !== false) this._scheduleRender();
      return Object.assign({}, this._viewport);
    }

    resetView() {
      this.orbit = { yaw: -0.72, pitch: -0.46, zoom: 1, panX: 0, panY: 0 };
      if (this.live) {
        this._liveFramingLocked = false;
        this._framing = null;
        this._ensureLiveFraming();
      }
      this._scheduleRender();
      return this;
    }

    _worldToView(value, center, scale) {
      const width = this._viewport.width;
      const height = this._viewport.height;
      const x = value[0] - center[0];
      const y = value[1] - center[1];
      const z = value[2] - center[2];
      const cy = Math.cos(this.orbit.yaw);
      const sy = Math.sin(this.orbit.yaw);
      const cp = Math.cos(this.orbit.pitch);
      const sp = Math.sin(this.orbit.pitch);
      const x1 = cy * x - sy * z;
      const z1 = sy * x + cy * z;
      const y2 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;
      return {
        x: width * 0.5 + this.orbit.panX + x1 * scale * this.orbit.zoom,
        y: height * 0.5 + this.orbit.panY - y2 * scale * this.orbit.zoom,
        z: z2,
      };
    }

    _allScenePoints() {
      const output = [...this.model.scene, ...this.model.target];
      for (const candidate of this.model.candidates) output.push(origin(candidate.pose));
      for (const reference of this.model.references) output.push(origin(reference.pose));
      for (const robot of [this.model.actualRobot, this.model.plannedRobot]) {
        if (!robot) continue;
        for (const link of robot.links) output.push(link[0], link[1]);
      }
      for (const values of Object.values(this.model.trajectory)) output.push(...values);
      if (this.model.graspArrow) {
        output.push(origin(this.model.graspArrow.pregrasp), origin(this.model.graspArrow.grasp));
      }
      if (this.model.collision) {
        output.push(
          this.model.collision.capsuleStart,
          this.model.collision.capsuleEnd,
          this.model.collision.scenePoint,
          this.model.collision.capsulePoint,
        );
      }
      return output.filter(Boolean);
    }

    _robustCloudPoints(values) {
      if (!Array.isArray(values) || values.length <= 16) return values || [];
      const bounds = [];
      for (let axis = 0; axis < 3; axis += 1) {
        const sorted = values.map(value => value[axis]).filter(finite).sort((a, b) => a - b);
        if (!sorted.length) return [];
        const lowIndex = Math.floor((sorted.length - 1) * 0.02);
        const highIndex = Math.ceil((sorted.length - 1) * 0.98);
        bounds.push([sorted[lowIndex], sorted[highIndex]]);
      }
      const trimmed = values.filter(value => value.every((coordinate, axis) => (
        coordinate >= bounds[axis][0] && coordinate <= bounds[axis][1]
      )));
      return trimmed.length >= 3 ? trimmed : values;
    }

    _fitLockedFraming() {
      const focus = [];
      if (this.model.actualRobot) {
        for (const link of this.model.actualRobot.links) focus.push(link[0], link[1]);
      }
      if (this.model.basePose) focus.push(origin(this.model.basePose));
      if (this.model.cameraPose) focus.push(origin(this.model.cameraPose));
      focus.push(...this._robustCloudPoints(this.model.target));
      const selected = this.model.candidates.find(candidate => (
        candidate.candidateId === this._selection.candidateId
      ));
      if (selected) focus.push(origin(selected.pose));
      if (this.model.collision) {
        focus.push(
          this.model.collision.capsuleStart,
          this.model.collision.capsuleEnd,
          this.model.collision.scenePoint,
          this.model.collision.capsulePoint,
        );
      }
      const values = focus.filter(Boolean).length >= 3
        ? focus.filter(Boolean)
        : this._allScenePoints();
      if (!values.length) return null;
      const low = [Infinity, Infinity, Infinity];
      const high = [-Infinity, -Infinity, -Infinity];
      for (const value of values) {
        for (let axis = 0; axis < 3; axis += 1) {
          low[axis] = Math.min(low[axis], value[axis]);
          high[axis] = Math.max(high[axis], value[axis]);
        }
      }
      const center = low.map((value, axis) => (value + high[axis]) * 0.5);
      const span = Math.max(...low.map((value, axis) => high[axis] - value), 0.28);
      return { center, span, source: "bundle_locked" };
    }

    _viewTransform() {
      const framing = this._framing || this._fitLockedFraming();
      if (!framing) return null;
      const center = framing.center;
      const span = framing.span;
      const scale = Math.min(this._viewport.width, this._viewport.height) * 0.80 / span;
      return { center, span, project: value => this._worldToView(value, center, scale) };
    }

    _line(a, b, color, width, dash) {
      const context = this.context;
      context.beginPath();
      if (typeof context.setLineDash === "function") context.setLineDash(dash || []);
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.strokeStyle = color;
      context.lineWidth = width;
      context.stroke();
      if (typeof context.setLineDash === "function") context.setLineDash([]);
    }

    _circle(value, radius, fill, stroke, width) {
      const context = this.context;
      context.beginPath();
      context.arc(value.x, value.y, radius, 0, Math.PI * 2);
      if (fill) { context.fillStyle = fill; context.fill(); }
      if (stroke) { context.strokeStyle = stroke; context.lineWidth = width || 1; context.stroke(); }
    }

    _drawFloor(project) {
      const base = this.model.basePose;
      if (!base) return;
      const extent = 0.6;
      const spacing = 0.1;
      for (let index = -6; index <= 6; index += 1) {
        const offset = index * spacing;
        const alpha = index === 0 ? "#3b515c" : this.options.grid;
        this._line(
          project(transformPoint(base, [-extent, offset, 0])),
          project(transformPoint(base, [extent, offset, 0])),
          alpha,
          index === 0 ? 1.4 : 0.7,
        );
        this._line(
          project(transformPoint(base, [offset, -extent, 0])),
          project(transformPoint(base, [offset, extent, 0])),
          alpha,
          index === 0 ? 1.4 : 0.7,
        );
      }
    }

    _drawAxes(project, matrix, length, label, width) {
      const context = this.context;
      const start = origin(matrix);
      const screen = project(start);
      for (let axis = 0; axis < 3; axis += 1) {
        const endpoint = [
          start[0] + matrix[0][axis] * length,
          start[1] + matrix[1][axis] * length,
          start[2] + matrix[2][axis] * length,
        ];
        this._line(screen, project(endpoint), AXIS_COLORS[axis], width || 1.6);
      }
      if (label) {
        context.fillStyle = "#d6e0e5";
        context.font = "10px ui-monospace, monospace";
        context.fillText(label, screen.x + 6, screen.y - 6);
      }
    }

    _drawCamera(project, matrix) {
      if (!matrix) return;
      const near = 0.13;
      const halfX = 0.075;
      const halfY = 0.052;
      const cameraOrigin = transformPoint(matrix, [0, 0, 0]);
      const corners = [
        [-halfX, -halfY, near],
        [halfX, -halfY, near],
        [halfX, halfY, near],
        [-halfX, halfY, near],
      ].map(value => transformPoint(matrix, value));
      for (const corner of corners) {
        this._line(project(cameraOrigin), project(corner), "#b899ff", 1.4);
      }
      for (let index = 0; index < corners.length; index += 1) {
        this._line(
          project(corners[index]),
          project(corners[(index + 1) % corners.length]),
          "#b899ff",
          1.2,
        );
      }
      this._drawAxes(project, matrix, 0.075, "camera pose", 1.8);
    }

    _drawClouds(project) {
      const context = this.context;
      const values = [];
      for (const value of this.model.scene) values.push({ value, target: false });
      for (const value of this.model.target) values.push({ value, target: true });
      values.sort((left, right) => project(left.value).z - project(right.value).z);
      for (const item of values) {
        const screen = project(item.value);
        context.globalAlpha = item.target ? 0.9 : 0.44;
        this._circle(
          screen,
          item.target ? 1.8 : 1.1,
          item.target ? this.options.targetPoint : this.options.scenePoint,
        );
      }
      context.globalAlpha = 1;
    }

    _drawRobot(project, robot, ghost) {
      if (!robot) return;
      const context = this.context;
      const color = ghost ? this.options.plannedRobot : this.options.actualRobot;
      const width = ghost ? 6 : 9;
      context.lineCap = "round";
      context.lineJoin = "round";
      for (const link of robot.links) {
        const start = project(link[0]);
        const end = project(link[1]);
        this._line(start, end, "#030608", width + 4, ghost ? [6, 5] : []);
        this._line(start, end, color, width, ghost ? [6, 5] : []);
      }
      for (const joint of robot.joints) {
        const screen = project(joint);
        this._circle(screen, ghost ? 4.5 : 6, ghost ? "rgba(111,157,255,.3)" : "#1a262c", color, 2);
      }
      this._drawGripper(project, robot, ghost);
      context.lineCap = "butt";
      context.lineJoin = "miter";
    }

    _drawGripper(project, robot, ghost) {
      if (!robot.links.length) return;
      const last = robot.links[robot.links.length - 1];
      const before = project(last[0]);
      const tip = project(last[1]);
      let dx = tip.x - before.x;
      let dy = tip.y - before.y;
      const length = Math.hypot(dx, dy) || 1;
      dx /= length;
      dy /= length;
      const px = -dy;
      const py = dx;
      const aperture = 12;
      const finger = 17;
      const color = ghost ? this.options.plannedRobot : "#f3f6f7";
      const dash = ghost ? [5, 4] : [];
      const left = { x: tip.x + px * aperture, y: tip.y + py * aperture };
      const right = { x: tip.x - px * aperture, y: tip.y - py * aperture };
      this._line(left, right, color, ghost ? 2 : 3, dash);
      this._line(left, { x: left.x + dx * finger, y: left.y + dy * finger }, color, ghost ? 2 : 3, dash);
      this._line(right, { x: right.x + dx * finger, y: right.y + dy * finger }, color, ghost ? 2 : 3, dash);
    }

    _drawTrajectory(project) {
      for (const [name, values] of Object.entries(this.model.trajectory)) {
        for (let index = 1; index < values.length; index += 1) {
          this._line(
            project(values[index - 1]),
            project(values[index]),
            PATH_COLORS[name] || "#ffffff",
            2.4,
          );
        }
      }
    }

    _drawArrow(project, start, end, color) {
      const a = project(start);
      const b = project(end);
      this._line(a, b, color, 3);
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const length = Math.hypot(dx, dy) || 1;
      dx /= length;
      dy /= length;
      const px = -dy;
      const py = dx;
      const back = 11;
      const side = 5;
      const first = { x: b.x - dx * back + px * side, y: b.y - dy * back + py * side };
      const second = { x: b.x - dx * back - px * side, y: b.y - dy * back - py * side };
      const context = this.context;
      context.beginPath();
      context.moveTo(b.x, b.y);
      context.lineTo(first.x, first.y);
      context.lineTo(second.x, second.y);
      context.closePath();
      context.fillStyle = color;
      context.fill();
    }

    _drawGrasps(project, span) {
      const showAll = this.options.showAllCandidates === true;
      for (const candidate of this.model.candidates) {
        const selected = candidate.candidateId === this._selection.candidateId
          || candidate.status === "selected";
        if (!showAll && !selected) continue;
        this._drawAxes(
          project,
          candidate.pose,
          span * (selected ? 0.075 : 0.045),
          selected ? `grasp pose #${candidate.candidateId}` : null,
          selected ? 2.6 : 1.1,
        );
      }
      if (this.model.graspArrow) {
        const pregrasp = origin(this.model.graspArrow.pregrasp);
        const grasp = origin(this.model.graspArrow.grasp);
        this._drawArrow(project, pregrasp, grasp, "#f0b758");
        this._drawAxes(project, this.model.graspArrow.pregrasp, span * 0.055, "pregrasp", 1.4);
        this._drawAxes(project, this.model.graspArrow.grasp, span * 0.065, "grasp", 2.2);
      }
    }

    _drawCollision(project, scale) {
      const collision = this.model.collision;
      if (!collision) return;
      const start = project(collision.capsuleStart);
      const end = project(collision.capsuleEnd);
      const radiusPixels = Math.max(7, Math.min(28, collision.radiusM * scale * this.orbit.zoom * 2));
      this.context.globalAlpha = 0.45;
      this._line(start, end, this.options.collision, radiusPixels);
      this.context.globalAlpha = 1;
      this._line(start, end, this.options.collision, 2);
      const scene = project(collision.scenePoint);
      const capsule = project(collision.capsulePoint);
      this._line(capsule, scene, this.options.witness, 3.2);
      this._circle(scene, 6, this.options.witness, "#ffffff", 1.4);
      this._circle(capsule, 5, "#ffffff", this.options.collision, 1.4);
      this.context.font = "10px ui-monospace, monospace";
      this.context.textAlign = "left";
      this.context.fillStyle = this.options.witness;
      this.context.fillText("scene nearest", scene.x + 8, scene.y - 7);
      this.context.fillStyle = this.options.collision;
      this.context.fillText("capsule nearest", capsule.x + 8, capsule.y + 13);
    }

    render() {
      if (this._destroyed) return false;
      try {
        const { width, height, ratio } = this._viewport;
        const context = this.context;
        if (typeof context.setTransform === "function") {
          context.setTransform(ratio, 0, 0, ratio, 0, 0);
        }
        context.clearRect(0, 0, width, height);
        context.fillStyle = this.options.background;
        context.fillRect(0, 0, width, height);
        const view = this._viewTransform();
        if (!view) {
          context.fillStyle = "#8fa0a8";
          context.font = "12px ui-monospace, monospace";
          context.textAlign = "center";
          context.fillText("No frame-consistent scene geometry", width * 0.5, height * 0.5);
        } else {
          const project = view.project;
          const pixelScale = Math.min(width, height) * 0.68 / view.span;
          this._drawFloor(project);
          this._drawColoredCloud(view);
          if (this.model.basePose) this._drawAxes(project, this.model.basePose, view.span * 0.075, "base", 2.2);
          this._drawCamera(project, this.model.cameraPose);
          this._drawClouds(project);
          this._drawTrajectory(project);
          this._drawRobot(project, this.model.plannedRobot, true);
          this._drawRobot(project, this.model.actualRobot, false);
          this._drawGrasps(project, view.span);
          this._drawCollision(project, pixelScale);
        }
        context.strokeStyle = "#243139";
        context.lineWidth = 1;
        context.strokeRect(0.5, 0.5, width - 1, height - 1);
        this._lastRenderMs = clockNow();
        this._renderCount += 1;
        return true;
      } catch (error) {
        this._diagnose("RENDER_ERROR", String(error && error.message || error), "render");
        return false;
      }
    }

    getDiagnostics() {
      return this.diagnostics.map(item => Object.assign({}, item));
    }

    getState() {
      return {
        version: VERSION,
        frame: this.displayFrame,
        live: this.live,
        reducedMotion: this.reducedMotion,
        maxFps: this.maxFps,
        renderCount: this._renderCount,
        pendingUpdate: this._pendingHandle !== null,
        overlayAllowed: this.model.overlayAllowed,
        selectedCandidateId: this._selection.candidateId,
        counts: {
          scenePoints: this.model.scene.length,
          targetPoints: this.model.target.length,
          candidates: this.model.candidates.length,
          actualLinks: this.model.actualRobot ? this.model.actualRobot.links.length : 0,
          plannedLinks: this.model.plannedRobot ? this.model.plannedRobot.links.length : 0,
          collisionWitness: this.model.collision ? 1 : 0,
          coloredCloudPoints: this.model.coloredCloud ? this.model.coloredCloud.count : 0,
          diagnostics: this.diagnostics.length,
        },
        framing: this._framing ? {
          center: this._framing.center.slice(),
          span: this._framing.span,
          source: this._framing.source,
        } : null,
        orbit: Object.assign({}, this.orbit),
        filters: {
          displayOnly: this._filterStats.displayOnly,
          cloudVoxelM: this._filterStats.cloudVoxelM,
          cloudUpdatesSuppressed: this._filterStats.cloudUpdatesSuppressed,
          clouds: Object.fromEntries(Object.entries(this._filterStats.clouds).map(
            ([name, values]) => [name, Object.assign({}, values)],
          )),
        },
      };
    }

    destroy() {
      if (this._destroyed) return;
      this._destroyed = true;
      this._cancelPending();
      if (this._observer) this._observer.disconnect();
      for (const remove of this._listeners.splice(0)) {
        try { remove(); } catch (_) { /* fail-soft teardown */ }
      }
      this.bundle = null;
      this.model = this._emptyModel();
      this._framing = null;
      this._cloudCanvas = null;
      this._cloudCtx = null;
      this._cloudImage = null;
      this._cloudU32 = null;
      this._cloudZ = null;
    }
  }

  const API = Object.freeze({
    VERSION,
    create(canvas, options) {
      return new RuntimeScene(canvas, options);
    },
    validateFrame,
    isSupported() {
      return typeof root.HTMLCanvasElement !== "undefined"
        || typeof root.OffscreenCanvas !== "undefined";
    },
  });

  root.ZManipScene = API;
}(typeof window !== "undefined" ? window : globalThis));
