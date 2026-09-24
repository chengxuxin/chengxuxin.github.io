// Stress test: random AMO commands across the demo's full drag ranges, plus walking and random hand
// reaches (IK), held 1.5-3.5 s each, smoothed like a drag. Counts falls and hand reach error.
//   node test/commands.mjs <n scripts> [seed]
import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { AMOPolicy } from '../src/policy.js';
import { AMOSim } from '../src/sim.js';

const N = +(process.argv[2] ?? 6), T = 40;
let seed = +(process.argv[3] ?? 1);
const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
const U = (lo, hi) => lo + (hi - lo) * rand();
const A = (p) => new URL('../../assets/amo/data/' + p, import.meta.url);
const buf = fs.readFileSync(A('amo.bin'));
const policy = new AMOPolicy(JSON.parse(fs.readFileSync(A('amo.json'))), buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const mujoco = await loadMujoco();
const vfs = new mujoco.MjVFS();
vfs.addBuffer('g1_amo.mjb', new Uint8Array(fs.readFileSync(A('g1_amo.mjb'))));
const sim = new AMOSim(mujoco, mujoco.MjModel.from_binary_path('g1_amo.mjb', vfs), policy);
sim.rng = rand;

let falls = 0; const reachErr = [];
for (let n = 0; n < N; n++) {
  sim.reset();
  const cmd = new Float32Array(8), tgt = new Float32Array(8);
  let next = 0, fell = false, measure = -1; const hand = [null, null];
  for (let k = 0; k < T / 0.02 && !fell; k++) {
    const t = k * 0.02;
    if (t >= next) {                                   // same ranges as the demo's drag / scroll / keys
      tgt[3] = U(-0.42, 0.02); tgt[5] = U(-0.45, 1.5); tgt[6] = U(-0.6, 0.6); tgt[4] = U(-1.4, 1.4);
      tgt[0] = rand() < 0.3 ? U(-0.3, 0.4) : 0; tgt[2] = rand() < 0.2 ? U(-0.3, 0.3) : 0;
      // hand targets in the pelvis heading frame: 15-60 cm ahead, out to the side, from knee to above the head
      const q = sim.d.qpos, h = Math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
      for (const side of [0, 1]) {
        if (rand() < 0.5) { hand[side] = null; sim.setHandTarget(side, null); continue; }
        const f = U(0.15, 0.6), l = (side ? -1 : 1) * U(0.05, 0.55), z = q[2] + U(-0.5, 0.7);
        hand[side] = [q[0] + f * Math.cos(h) - l * Math.sin(h), q[1] + f * Math.sin(h) + l * Math.cos(h), z];
        sim.setHandTarget(side, hand[side]);
      }
      measure = t + 1.5;
      next = t + U(1.5, 3.5);
    }
    for (const i of [0, 2, 3, 4, 5, 6]) cmd[i] += (tgt[i] - cmd[i]) * 0.04;
    sim.commands.set(cmd);
    sim.step();
    if (Math.abs(t - measure) < 0.01)                  // how close each reaching hand got, 1.5 s after its target was set
      for (const side of [0, 1]) if (hand[side]) { const p = sim.point(side + 1); reachErr.push(Math.hypot(p[0] - hand[side][0], p[1] - hand[side][1], p[2] - hand[side][2])); }
    if (sim.d.qpos[2] < 0.3) { fell = true; console.log(`  script ${n}: fell at ${t.toFixed(1)} s  (h ${(0.75 + cmd[3]).toFixed(2)} pitch ${cmd[5].toFixed(2)} roll ${cmd[6].toFixed(2)} yaw ${cmd[4].toFixed(2)} vx ${cmd[0].toFixed(2)} hands ${hand.map((x) => (x ? 'on' : '-')).join(' ')})`); }
  }
  falls += fell;
}
reachErr.sort((a, b) => a - b);
const pct = (p) => (reachErr[Math.floor(p * (reachErr.length - 1))] * 100).toFixed(1);
console.log(`${N - falls}/${N} random ${T} s command scripts without a fall; hand reach error after 1.5 s: median ${pct(0.5)} cm, 90% ${pct(0.9)} cm (${reachErr.length} reaches; far targets can be out of reach)`);
