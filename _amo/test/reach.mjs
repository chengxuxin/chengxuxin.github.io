// Hand reaching accuracy on targets that are reachable by construction (the hand position of a random
// arm pose, computed on the current body), robot standing.   node test/reach.mjs [n]
import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { AMOPolicy } from '../src/policy.js';
import { AMOSim } from '../src/sim.js';

const N = +(process.argv[2] ?? 12);
let seed = 3; const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
const A = (p) => new URL('../../assets/amo/data/' + p, import.meta.url);
const buf = fs.readFileSync(A('amo.bin'));
const policy = new AMOPolicy(JSON.parse(fs.readFileSync(A('amo.json'))), buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
vfs.addBuffer('g1_amo.mjb', new Uint8Array(fs.readFileSync(A('g1_amo.mjb'))));
const sim = new AMOSim(mujoco, mujoco.MjModel.from_binary_path('g1_amo.mjb', vfs), policy);
for (let k = 0; k < 100; k++) sim.step();                // settle
const err = [];
for (let n = 0; n < N; n++) {
  const side = n % 2, kd = sim.kd;
  kd.qpos.set(sim.d.qpos);
  const pose = [rand() * 2.4 - 1.6, (side ? -1 : 1) * rand() * 1.2, rand() * 1.6 - 0.8, rand() * 1.6];   // pitch, roll (outward), yaw, elbow
  for (let j = 0; j < 4; j++) kd.qpos[22 + side * 4 + j] = pose[j];
  mujoco.mj_kinematics(sim.m, kd);
  const T = sim.point(side + 1, kd);
  sim.setHandTarget(side, T);
  const trace = [];
  for (let k = 0; k < 100; k++) { sim.step(); if (k % 25 === 24) { const p = sim.point(side + 1); trace.push(Math.hypot(p[0] - T[0], p[1] - T[1], p[2] - T[2])); } }
  err.push(trace.at(-1));
  console.log(`reach ${n} (${side ? 'right' : 'left'}): error at 0.5/1/1.5/2 s = ${trace.map((e) => (e * 100).toFixed(1)).join(' / ')} cm`);
  sim.setHandTarget(side, null);
}
err.sort((a, b) => a - b);
console.log(`median ${(err[err.length >> 1] * 100).toFixed(1)} cm, worst ${(err.at(-1) * 100).toFixed(1)} cm; pelvis z ${sim.d.qpos[2].toFixed(2)}`);
