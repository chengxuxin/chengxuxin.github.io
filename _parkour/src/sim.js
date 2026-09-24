// MuJoCo (WASM) sim2sim of the Isaaclab_Parkour Go2 teacher: a line-by-line port of py/go2_env.py.
// The model is compiled once with a fixed-size heightfield; each new track only rewrites hfield_data.
import { HS } from './terrain.js';

const ISAAC2MJ = []; for (let t = 0; t < 3; t++) for (let leg = 0; leg < 4; leg++) ISAAC2MJ.push(leg * 3 + t);
const DEFAULT = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5];
const KP = 40, KD = 1;
const EFFORT = [35, 35, 35, 35, 40, 40, 40, 40, 40, 40, 40, 40];
const SAT = [35, 35, 35, 35, 45, 45, 45, 45, 45, 45, 45, 45];
const VLIM = [52.4, 52.4, 52.4, 52.4, 30.1, 30.1, 30.1, 30.1, 30.1, 30.1, 30.1, 30.1];
const TERRAIN = { friction: 1.2, solref: '0.005 1' };      // matches the robot's stiff contacts (see go2_env CONTACT.stiff)

// height-scan grid: Isaac grid_pattern, 'xy' ordering (x fastest), offset (0.375, 0)
export const SCAN = [];
for (let iy = 0; iy < 11; iy++) for (let ix = 0; ix < 12; ix++) SCAN.push([-0.825 + ix * 0.15 + 0.375, -0.75 + iy * 0.15]);

const wrap = (a) => { const T = 2 * Math.PI; let m = (a + Math.PI) % T; if (m < 0) m += T; return m - Math.PI; };
const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
export function quatToRpy(w, x, y, z) {
  return [Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
    Math.asin(clip(2 * (w * y - z * x), -1, 1)),
    Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))];
}

// Heightfield samples covering a track + border (rows = y, cols = x), like Track.hfield() in Python.
export function hfieldGrid(track, res, border = 2) {
  const xMin = track.x0 - border, xMax = track.x0 + track.length + border, yMin = -2 - border, yMax = 2 + border;
  const nx = Math.round((xMax - xMin) / res) + 1, ny = Math.round((yMax - yMin) / res) + 1;
  const Z = new Float64Array(nx * ny);
  let lo = Infinity, hi = -Infinity;
  for (let r = 0; r < ny; r++) {
    const y = r === ny - 1 ? yMax : yMin + r * ((yMax - yMin) / (ny - 1));
    for (let c = 0; c < nx; c++) {
      const x = c === nx - 1 ? xMax : xMin + c * ((xMax - xMin) / (nx - 1));
      const z = track.height(x, y); Z[r * nx + c] = z; lo = Math.min(lo, z); hi = Math.max(hi, z);
    }
  }
  return { Z, nx, ny, xMin, xMax, yMin, yMax, lo, hi };
}

export class Go2Parkour {
  // opts.zRange: fixed [zmin, zmax] of the heightfield (live tracks); omitted = Python's per-track fit
  constructor(mujoco, physXml, policy, track, { vx = 0.8, substeps = 10, res = 0.04, zRange = null, parkourFlag = true } = {}) {
    Object.assign(this, { mujoco, policy, vx, substeps, res, zRange });
    this.flag = parkourFlag ? [1, 0] : [0, 1];
    const g = hfieldGrid(track, res);
    const [zmin, zmax] = this._zr(g);
    const world = `
  <asset><hfield name="terrain" nrow="${g.ny}" ncol="${g.nx}" size="${(g.xMax - g.xMin) / 2} ${(g.yMax - g.yMin) / 2} ${zmax - zmin} 0.5"/></asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="${(g.xMax + g.xMin) / 2} ${(g.yMax + g.yMin) / 2} ${zmin}" friction="${TERRAIN.friction} 0.02 0.01" solref="${TERRAIN.solref}"/>
  </worldbody>`;
    this.m = mujoco.MjModel.from_xml_string(physXml.replace('</mujoco>', world + '\n</mujoco>'), new mujoco.MjVFS());
    this.m.opt.timestep = 0.02 / substeps;
    this.d = new mujoco.MjData(this.m);
    this.shape = [g.ny, g.nx];
    this.terrainGeom = this.m.ngeom - 1;
    this.prop = new Float64Array(53); this.scan = new Float64Array(132); this.histIn = new Float64Array(530);
    this.hist = new Float64Array(530); this.target = new Float64Array(12); this.lastAction = new Float64Array(12);
    this.setTrack(track, g);
  }

  _zr(g) { return this.zRange ?? [g.lo - 0.05, Math.max(g.hi, g.lo + 0.1)]; }

  // Swap in a new track of the same footprint: rewrite the heightfield, then reset.
  setTrack(track, g = hfieldGrid(track, this.res)) {
    if (g.ny !== this.shape[0] || g.nx !== this.shape[1]) throw new Error('track footprint changed');
    const [zmin, zmax] = this._zr(g), hd = this.m.hfield_data;
    if (g.lo < zmin || g.hi > zmax) console.warn('terrain exceeds the heightfield range', g.lo, g.hi);
    for (let k = 0; k < g.Z.length; k++) hd[k] = (g.Z[k] - zmin) / (zmax - zmin);
    this.track = track;
    this.reset();
  }

  reset(start = [-7, 0]) {
    const { m, d, mujoco } = this;
    mujoco.mj_resetData(m, d);
    const q = d.qpos;
    q[0] = start[0]; q[1] = start[1]; q[2] = 0.4 + this.track.height(start[0], start[1]);
    q[3] = 1; q[4] = 0; q[5] = 0; q[6] = 0;
    for (let k = 0; k < 12; k++) q[7 + ISAAC2MJ[k]] = DEFAULT[k];
    mujoco.mj_forward(m, d);
    this.lastAction.fill(0); this.hist.fill(0);
    this.epLen = 0; this.counter = 0; this.sub = 0; this.goalIdx = 0; this.goalTimer = 0;
    const G = this.track.goals;
    this.envGoals = [...G, G[G.length - 1], G[G.length - 1]];
    this.deltaYaw = this.deltaNextYaw = 0;
    this.scan.fill(0);
    this.prevContact = [false, false, false, false];
    for (let k = 0; k < 12; k++) this.target[ISAAC2MJ[k]] = DEFAULT[k];
    this._updateGoals();
  }

  get finished() { return this.goalIdx >= this.envGoals.length - 2; }     // reached the last real goal

  _updateGoals() {
    const q = this.d.qpos, x = q[0], y = q[1];
    let cur = this.envGoals[this.goalIdx], nxt = this.envGoals[this.goalIdx + 1];
    if (Math.hypot(x - cur[0], y - cur[1]) < 0.2) this.goalTimer += 1;
    if (this.goalTimer > 0.1 / 0.02) {
      this.goalIdx = Math.min(this.goalIdx + 1, this.envGoals.length - 2); this.goalTimer = 0;
      cur = this.envGoals[this.goalIdx]; nxt = this.envGoals[this.goalIdx + 1];
    }
    this.targetYaw = Math.atan2(cur[1] - y, cur[0] - x);
    this.nextTargetYaw = Math.atan2(nxt[1] - y, nxt[0] - x);
  }

  _obs() {
    const d = this.d, q = d.qpos, qd = d.qvel, P = this.prop;
    const [roll, pitch, yaw] = quatToRpy(q[3], q[4], q[5], q[6]);
    if (this.counter % 5 === 0) {
      this.deltaYaw = this.targetYaw - wrap(yaw);
      this.deltaNextYaw = this.nextTargetYaw - wrap(yaw);
      const c = Math.cos(yaw), s = Math.sin(yaw);
      for (let k = 0; k < 132; k++) {
        const [sx, sy] = SCAN[k];
        const px = q[0] + c * sx - s * sy, py = q[1] + s * sx + c * sy;
        this.scan[k] = clip(q[2] - this.track.height(px, py) - 0.3, -1, 1);
      }
    }
    const sd = d.sensordata, fill = [];
    for (let f = 0; f < 4; f++) { const c = sd[f] > 2.0; fill.push((c || this.prevContact[f] ? 1 : 0) - 0.5); this.prevContact[f] = c; }
    P[0] = qd[3] * 0.25; P[1] = qd[4] * 0.25; P[2] = qd[5] * 0.25;
    P[3] = wrap(roll); P[4] = wrap(pitch); P[5] = 0; P[6] = this.deltaYaw; P[7] = this.deltaNextYaw;
    P[8] = 0; P[9] = 0; P[10] = this.vx; P[11] = this.flag[0]; P[12] = this.flag[1];
    for (let k = 0; k < 12; k++) {
      const j = ISAAC2MJ[k];
      P[13 + k] = q[7 + j] - DEFAULT[k];
      P[25 + k] = qd[6 + j] * 0.05;
      P[37 + k] = this.lastAction[k];
    }
    for (let f = 0; f < 4; f++) P[49 + f] = fill[f];
    this.histIn.set(this.hist);
    const H = this.hist;
    if (this.epLen <= 1) { for (let t = 0; t < 10; t++) H.set(P, t * 53); }
    else { H.copyWithin(0, 53); H.set(P, 9 * 53); }
    for (let t = this.epLen <= 1 ? 0 : 9; t < 10; t++) { H[t * 53 + 6] = 0; H[t * 53 + 7] = 0; }
  }

  // One 50 Hz control step.
  step() { for (let s = 0; s < this.substeps; s++) this.tick(); }

  // One physics substep (2 ms); the policy runs at the start of every control period. Lets the
  // page advance by wall-clock time with smooth poses, identical to calling step().
  tick() {
    const { m, d, mujoco } = this;
    if (this.sub === 0) {
      this._obs();
      const a = this.policy.act(this.prop, this.scan, this.histIn);
      this.lastAction.set(a);
      for (let k = 0; k < 12; k++) this.target[ISAAC2MJ[k]] = clip(a[k], -4.8, 4.8) * 0.25 + DEFAULT[k];
    }
    const q = d.qpos, qd = d.qvel, ctrl = d.ctrl;
    for (let k = 0; k < 12; k++) {
      const j = ISAAC2MJ[k], v = qd[6 + j];
      const tau = KP * (this.target[j] - q[7 + j]) - KD * v;
      const mx = clip(SAT[k] * (1.0 - v / VLIM[k]), 0, EFFORT[k]), mn = clip(SAT[k] * (-1.0 - v / VLIM[k]), -EFFORT[k], 0);
      ctrl[j] = clip(tau, mn, mx);
    }
    mujoco.mj_step(m, d);
    if (++this.sub === this.substeps) {
      this.sub = 0; this.epLen += 1; this.counter += 1;
      this._updateGoals();
      return true;                                   // a control step just completed
    }
    return false;
  }
}

export { ISAAC2MJ, HS };
