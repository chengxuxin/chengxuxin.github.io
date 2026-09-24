// AMO controller on MuJoCo (WASM): a line-by-line port of play_amo.py's HumanoidEnv (via py/amo_env.py).
// commands = [vx, target yaw, vy, height offset (+0.75 m), torso yaw, torso pitch, torso roll, arms]
const KP = [150, 150, 150, 300, 80, 20, 150, 150, 150, 300, 80, 20, 400, 400, 400, 80, 80, 40, 60, 80, 80, 40, 60];
const KD = [2, 2, 2, 4, 2, 1, 2, 2, 2, 4, 2, 1, 15, 15, 15, 2, 2, 1, 1, 2, 2, 1, 1];
export const DEFAULT = [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0, 0.5, 0, 0.2, 0.3, 0.5, 0, -0.2, 0.3];
const TAU = [88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 25, 25, 25];
const f32 = Math.fround, TWO_PI = 2 * Math.PI, VEL_SCALE = f32(0.05);
// drag points (body, offset in the body frame): chest, left hand, right hand (centres of the torso / hand meshes)
export const HANDLES = [['torso_link', [0.02, 0, 0.17]], ['left_rubber_hand', [0.066, -0.015, 0.01]], ['right_rubber_hand', [0.066, 0.015, 0.01]]];
const ARM_STEP = 0.08;                               // max arm target change per control step (4 rad/s)
// natural arm ranges for reaching (inside the joint limits; the arms have no collision geometry, so the
// shoulder may not swing inward into the torso): shoulder pitch, roll (outward +), yaw, elbow
const REACH = [[-2.6, 1.0], [-0.15, 2.0], [-1.4, 1.4], [-0.9, 2.0]];
const solve4 = (A, b) => {                           // Gaussian elimination with partial pivoting
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 4; c++) {
    let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < 4; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k < 5; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = [0, 0, 0, 0];
  for (let r = 3; r >= 0; r--) { let v = M[r][4]; for (let k = r + 1; k < 4; k++) v -= M[r][k] * x[k]; x[r] = v / M[r][r]; }
  return x;
};
const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const pymod = (a, n) => { const r = a % n; return r !== 0 && (r < 0) !== (n < 0) ? r + n : r; };   // np.remainder

// quatToEuler from play_amo.py; its inputs are float32, so its arithmetic runs in float32
function rpyF32(w, x, y, z) {
  const sinr = f32(2 * f32(f32(w * x) + f32(y * z))), cosr = f32(1 - f32(2 * f32(f32(x * x) + f32(y * y))));
  const sinp = f32(2 * f32(f32(w * y) - f32(z * x)));
  const siny = f32(2 * f32(f32(w * z) + f32(x * y))), cosy = f32(1 - f32(2 * f32(f32(y * y) + f32(z * z))));
  return [f32(Math.atan2(sinr, cosr)), Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : f32(Math.asin(sinp)), f32(Math.atan2(siny, cosy))];
}

export class AMOSim {
  constructor(mujoco, model, policy) {
    Object.assign(this, { mujoco, m: model, policy });
    model.opt.timestep = 0.002;
    this.d = new mujoco.MjData(model);
    const sensor = (n) => model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR.value, n)];
    this.sQuat = sensor('orientation'); this.sGyro = sensor('angular-velocity');
    this.commands = new Float32Array(8);
    this.dofPos = new Float64Array(23); this.dofVel = new Float64Array(23); this.quat = new Float64Array(4); this.angVel = new Float64Array(3);
    this.prop = new Float64Array(93); this.obs = new Float64Array(1043); this.extraIn = new Float64Array(2325);
    this.hist = new Float64Array(930); this.extra = new Float64Array(2325);
    this.pd = new Float64Array(23); this.lastAction = new Float64Array(23); this.x12 = new Float64Array(12);
    this.armAction = new Float64Array(8); this.prevArm = new Float64Array(8);
    this.rng = Math.random;                                         // random arm targets (commands[7])
    // hands: IK on a scratch MjData; the arms (4 dof each) then hold whatever pose reached the target
    this.kd = new mujoco.MjData(model);
    this.handles = HANDLES.map(([n, off]) => [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, n), off]);
    this.armRange = [];
    for (let k = 0; k < 8; k++) {
      const j = 16 + k, [lo, hi] = REACH[k % 4], out = k >= 4 && k % 4 === 1 ? -1 : 1;    // right roll: outward is -
      const [a, b] = out > 0 ? [lo, hi] : [-hi, -lo];
      this.armRange.push([Math.max(a, model.jnt_range[2 * j]), Math.min(b, model.jnt_range[2 * j + 1])]);
    }
    this.handTarget = [null, null]; this.handBias = [[0, 0, 0], [0, 0, 0]]; this.handRelease = [Infinity, Infinity];
    this.reset();
  }

  reset() {
    const { mujoco, m, d } = this;
    mujoco.mj_resetDataKeyframe(m, d, 0);
    mujoco.mj_step(m, d);
    this.i = 0;
    this.lastAction.fill(0); this.hist.fill(0); this.extra.fill(0);
    for (let k = 0; k < 8; k++) this.armAction[k] = this.prevArm[k] = DEFAULT[15 + k];
    this.armBlend = 0; this.toggleArm = false;
    this.gait = [0.25, 0.25]; this.stand = true;
    if (this.handTarget) this.handTarget = [null, null];
  }

  // Blend the arms to new joint targets over 2 s (8: left shoulder pitch/roll/yaw, elbow, right ...).
  setArms(target) {
    for (let k = 0; k < 8; k++) { this.prevArm[k] = this.dofPos[15 + k]; this.armAction[k] = target[k]; }
    this.armBlend = 0;
  }

  // Move one hand (0 left, 1 right) to a world-space point. Releasing (null) keeps reaching for the last
  // point for up to 1.5 s, so a quick flick still lands, then the arm holds its pose.
  setHandTarget(side, p) {
    if (!p) { if (this.handTarget[side]) this.handRelease[side] = this.i + 750; return; }
    if (!this.handTarget[side]) this.handBias[side] = [0, 0, 0];
    this.handTarget[side] = Float64Array.from(p); this.handRelease[side] = Infinity;
  }

  // world position of drag point h (0 chest, 1 left hand, 2 right hand) in MjData `d`
  point(h, d = this.d, out = [0, 0, 0]) {
    const [b, o] = this.handles[h], p = d.xpos, R = d.xmat;
    for (let i = 0; i < 3; i++) out[i] = p[b * 3 + i] + R[b * 9 + i * 3] * o[0] + R[b * 9 + i * 3 + 1] * o[1] + R[b * 9 + i * 3 + 2] * o[2];
    return out;
  }

  // Damped least squares on the 4 arm joints, from the current arm targets, against the current body
  // pose. The arm's PD sags under gravity, so the kinematic target is nudged by the measured hand error
  // (a slow integral in task space, capped at 12 cm).
  _reach(side) {
    const { mujoco, m, kd } = this, a = 22 + side * 4, h = side + 1, B = this.handBias[side], act = this.point(h);
    for (let i = 0; i < 3; i++) B[i] += 0.1 * (this.handTarget[side][i] - act[i]);
    const bn = Math.hypot(...B); if (bn > 0.12) for (let i = 0; i < 3; i++) B[i] *= 0.12 / bn;
    const T = [0, 1, 2].map((i) => this.handTarget[side][i] + B[i]);
    const q = Array.from(this.armAction.subarray(side * 4, side * 4 + 4)), p = [0, 0, 0], pj = [0, 0, 0], J = [];
    kd.qpos.set(this.d.qpos);
    for (let it = 0; it < 10; it++) {
      for (let j = 0; j < 4; j++) kd.qpos[a + j] = q[j];
      mujoco.mj_kinematics(m, kd); this.point(h, kd, p);
      const e = [T[0] - p[0], T[1] - p[1], T[2] - p[2]];
      if (Math.hypot(...e) < 0.004) break;
      for (let j = 0; j < 4; j++) {                                 // Jacobian by finite differences
        kd.qpos[a + j] = q[j] + 1e-4; mujoco.mj_kinematics(m, kd); this.point(h, kd, pj); kd.qpos[a + j] = q[j];
        J[j] = [(pj[0] - p[0]) / 1e-4, (pj[1] - p[1]) / 1e-4, (pj[2] - p[2]) / 1e-4];
      }
      // dq = (J^T J + (l^2 + m^2) I)^-1 (J^T e + m^2 (q_rest - q)): damped, and drawn toward AMO's default arm pose
      const A = [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => J[r][0] * J[c][0] + J[r][1] * J[c][1] + J[r][2] * J[c][2] + (r === c ? 0.007 : 0)));
      const g = [0, 1, 2, 3].map((r) => J[r][0] * e[0] + J[r][1] * e[1] + J[r][2] * e[2] + 0.001 * (DEFAULT[15 + side * 4 + r] - q[r]));
      const dq = solve4(A, g);
      for (let j = 0; j < 4; j++) {
        const [lo, hi] = this.armRange[side * 4 + j];
        q[j] = clip(q[j] + clip(dq[j], -0.3, 0.3), lo, hi);
      }
    }
    for (let j = 0; j < 4; j++) {                                   // rate-limited, and held (no blending)
      const k = side * 4 + j, v = this.armAction[k] + clip(q[j] - this.armAction[k], -ARM_STEP, ARM_STEP);
      this.armAction[k] = this.prevArm[k] = v;
    }
  }

  extract() {
    const d = this.d, q = d.qpos, v = d.qvel, s = d.sensordata;
    for (let k = 0; k < 23; k++) { this.dofPos[k] = f32(q[7 + k]); this.dofVel[k] = f32(v[6 + k]); }
    for (let k = 0; k < 4; k++) this.quat[k] = f32(s[this.sQuat + k]);
    for (let k = 0; k < 3; k++) this.angVel[k] = f32(s[this.sGyro + k]);
  }

  observe() {
    const c = this.commands, P = this.prop, O = this.obs, qa = this.quat;
    const rpy = rpyF32(qa[0], qa[1], qa[2], qa[3]);
    let dyaw = pymod(rpy[2] - c[1] + Math.PI, TWO_PI) - Math.PI;
    if (this.stand) dyaw = 0;
    const h = f32(0.75 + c[3]);
    const x = this.x12;
    x[0] = h; x[1] = c[4]; x[2] = c[5]; x[3] = c[6];
    for (let k = 0; k < 8; k++) x[4 + k] = this.dofPos[15 + k];
    const ad = this.policy.adapter(x);
    for (let k = 0; k < 3; k++) P[k] = f32(this.angVel[k] * 0.25);
    P[3] = rpy[0]; P[4] = rpy[1]; P[5] = Math.sin(dyaw); P[6] = Math.cos(dyaw);
    for (let k = 0; k < 23; k++) { P[7 + k] = this.dofPos[k] - DEFAULT[k]; P[30 + k] = f32(this.dofVel[k] * VEL_SCALE); P[53 + k] = this.lastAction[k]; }
    P[76] = Math.sin(this.gait[0] * TWO_PI); P[77] = Math.sin(this.gait[1] * TWO_PI);
    for (let k = 0; k < 15; k++) P[78 + k] = ad[k];
    // obs = [prop | demo | priv | hist (before this frame)]
    O.set(P, 0);
    O.fill(0, 93, 113);
    for (let k = 0; k < 8; k++) O[93 + k] = this.dofPos[15 + k];
    O[101] = c[0]; O[102] = c[2];
    this.stand = Math.abs(c[0]) < 0.1;
    O[104] = c[4]; O[105] = c[5]; O[106] = c[6]; O[107] = O[108] = O[109] = h;
    O.set(this.hist, 113);
    this.hist.copyWithin(0, 93); this.hist.set(P, 837);
    this.extra.copyWithin(0, 93); this.extra.set(P, 2232);
  }

  control() {
    this.observe();
    const O = this.obs, E = this.extraIn, c = this.commands;
    for (let k = 0; k < 1043; k++) O[k] = f32(O[k]);                // the released model runs in float32
    for (let k = 0; k < 2325; k++) E[k] = f32(this.extra[k]);
    const raw = this.policy.act(O, E);
    for (let k = 0; k < 15; k++) { raw[k] = clip(raw[k], -40, 40); this.lastAction[k] = raw[k]; }
    for (let k = 0; k < 8; k++) this.lastAction[15 + k] = (this.dofPos[15 + k] - DEFAULT[15 + k]) / 0.25;
    if (this.i % 300 === 0 && this.i > 0 && c[7]) {
      this.armBlend = 0;
      for (let k = 0; k < 8; k++) { this.prevArm[k] = this.dofPos[15 + k]; this.armAction[k] = this.rng() * 0.8 - 0.4; }
      this.toggleArm = true;
    } else if (!c[7] && this.toggleArm) {
      this.toggleArm = false; this.armBlend = 0;
      for (let k = 0; k < 8; k++) { this.prevArm[k] = this.dofPos[15 + k]; this.armAction[k] = DEFAULT[15 + k]; }
    }
    for (let side = 0; side < 2; side++) {
      if (this.handTarget[side] && this.i >= this.handRelease[side]) this.handTarget[side] = null;
      if (this.handTarget[side]) this._reach(side);
    }
    for (let k = 0; k < 15; k++) this.pd[k] = raw[k] * 0.25 + DEFAULT[k];
    for (let k = 0; k < 8; k++) this.pd[15 + k] = (1 - this.armBlend) * this.prevArm[k] + this.armBlend * this.armAction[k];
    this.armBlend = Math.min(1, this.armBlend + 0.01);
    let g = [pymod(this.gait[0] + 0.02 * 1.3, 1), pymod(this.gait[1] + 0.02 * 1.3, 1)];
    if (this.stand && (Math.abs(g[0] - 0.25) < 0.05 || Math.abs(g[1] - 0.25) < 0.05)) g = [0.25, 0.25];
    if (!this.stand && Math.abs(g[0] - 0.25) < 0.05 && Math.abs(g[1] - 0.25) < 0.05) g = [0.25, 0.75];
    this.gait = g;
  }

  // one 500 Hz physics substep; the policy runs every 10th
  substep() {
    const { mujoco, m, d } = this;
    this.extract();
    if (this.i % 10 === 0) this.control();
    const ctrl = d.ctrl;
    for (let k = 0; k < 23; k++) ctrl[k] = clip((this.pd[k] - this.dofPos[k]) * KP[k] - this.dofVel[k] * KD[k], -TAU[k], TAU[k]);
    mujoco.mj_step(m, d);
    this.i++;
    return this.i % 10 === 0;
  }

  step() { for (let k = 0; k < 10; k++) this.substep(); }
}
