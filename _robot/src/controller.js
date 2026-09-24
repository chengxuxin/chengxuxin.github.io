// GMT sim2sim control loop on MuJoCo WASM (port of HumanoidEnv.run).
// Physics 1 kHz, policy 50 Hz, PD torque control computed in JS.

import { MotionRef } from './motion.js';

const KP = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 150, 150, 150, 40, 40, 40, 40, 40, 40, 40, 40];
const KD = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5];
export const Q0 = [-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0, 0, 0, 0, 0, 0.4, 0, 1.2, 0, -0.4, 0, 1.2];
const TL = [88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 25, 25, 25];
const DECIMATION = 20;

// MuJoCo quats are wxyz (port of sim2sim.quatToEuler)
function quatToRollPitch(q) {
  const [w, x, y, z] = q;
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const sinp = 2 * (w * y - z * x);
  const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
  return [roll, pitch];
}

export class GMTController {
  constructor(mujoco, model, data, policy) {
    Object.assign(this, { mujoco, model, data, policy });
    this.obs = new Float32Array(2154);
    this.prop = new Float32Array(74);
    this.hist = [];
    this.lastAction = new Float32Array(23);
    this.pd = Float64Array.from(Q0);
    this.f32 = new Float32Array(1);
  }

  setMotion(motion) {       // MotionRef; restarts reference time, keeps the robot state
    this.motion = motion;
    this.step = 0;
  }

  reset() {
    const { mujoco, model, data } = this;
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_step(model, data);
    this.hist = Array.from({ length: 20 }, () => new Float32Array(74));
    this.lastAction.fill(0);
    this.pd = Float64Array.from(Q0);
    this.i = 0; this.step = 0;
  }

  control() {
    const { data, prop, obs } = this;
    const qpos = data.qpos, qvel = data.qvel, sd = data.sensordata;
    this.motion.mimicObs(this.step, obs);
    const [roll, pitch] = quatToRollPitch([sd[0], sd[1], sd[2], sd[3]]);
    prop[0] = sd[7] * 0.25; prop[1] = sd[8] * 0.25; prop[2] = sd[9] * 0.25;
    prop[3] = roll; prop[4] = pitch;
    for (let k = 0; k < 23; k++) {
      prop[5 + k] = Math.fround(qpos[7 + k]) - Q0[k];
      const dv = (k === 4 || k === 5 || k === 10 || k === 11) ? 0 : Math.fround(qvel[6 + k]);
      prop[28 + k] = dv * 0.05;
      prop[51 + k] = this.lastAction[k];
    }
    obs.set(prop, 600);
    for (let h = 0; h < 20; h++) obs.set(this.hist[h], 674 + h * 74);

    const raw = this.policy.act(obs);
    this.lastAction.set(raw);
    for (let k = 0; k < 23; k++) this.pd[k] = Math.max(-10, Math.min(10, raw[k])) * 0.5 + Q0[k];
    this.hist.shift(); this.hist.push(Float32Array.from(prop));
    this.step++;
  }

  // One 1 ms physics tick
  tick() {
    const { mujoco, model, data } = this;
    if (this.i % DECIMATION === 0) this.control();
    const qpos = data.qpos, qvel = data.qvel, ctrl = data.ctrl;
    for (let k = 0; k < 23; k++) {
      const tau = (this.pd[k] - Math.fround(qpos[7 + k])) * KP[k] - Math.fround(qvel[6 + k]) * KD[k];
      ctrl[k] = Math.max(-TL[k], Math.min(TL[k], tau));
    }
    mujoco.mj_step(model, data);
    this.i++;
  }
}

export { MotionRef };
