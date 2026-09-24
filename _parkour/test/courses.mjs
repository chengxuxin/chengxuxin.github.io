// Course completion rate at a given commanded speed, using the page's own course sampling.
//   node test/courses.mjs <vx> <n courses> [first seed]   (npm test)
import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { TeacherPolicy } from '../src/policy.js';
import { Go2Parkour, quatToRpy } from '../src/sim.js';
import { TYPES, makeCourse, stuckAfter } from '../src/course.js';

const vx = +process.argv[2], N = +process.argv[3], seed0 = +(process.argv[4] ?? 1000);
const A = (p) => new URL('../../assets/parkour/data/' + p, import.meta.url);
const bin = fs.readFileSync(A('policy.bin'));
const policy = new TeacherPolicy(JSON.parse(fs.readFileSync(A('policy.json'))), new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4));
const mujoco = await loadMujoco();
const env = new Go2Parkour(mujoco, fs.readFileSync(A('go2_phys.xml'), 'utf8'), policy, makeCourse(seed0), { zRange: [-1.2, 1.8], vx });

const res = { done: 0, fell: 0, stuck: 0 }, failAt = {}, times = [];
for (let e = 0; e < N; e++) {
  const track = e ? makeCourse(seed0 + e) : env.track; if (e) env.setTrack(track);
  let s = 0, last = 0, tGoal = 0, end = '';
  for (; s < 15000; s++) {
    env.step();
    const q = env.d.qpos, [r, p] = quatToRpy(q[3], q[4], q[5], q[6]);
    if (env.goalIdx !== last) { last = env.goalIdx; tGoal = s; }
    if (env.finished) { end = 'done'; break; }
    if (Math.abs(r) > 1.2 || Math.abs(p) > 1.2) { end = 'fell'; break; }
    if ((s - tGoal) * 0.02 > stuckAfter(vx)) { end = 'stuck'; break; }
  }
  end ||= 'stuck';
  res[end]++;
  if (end === 'done') times.push(s * 0.02);
  else {                                                   // which terrain it failed on
    const x = env.d.qpos[0], k = track.spans.findIndex(([a, b]) => x < b + 0.8);
    const t = TYPES[track.subs[Math.max(0, k)].type].label + ` ${track.subs[Math.max(0, k)].difficulty.toFixed(2)}`;
    (failAt[end] ||= []).push(t);
  }
}
const avg = times.length ? times.reduce((a, b) => a + b, 0) / times.length : 0;
console.log(`vx ${vx.toFixed(2)}: ${res.done}/${N} done, ${res.fell} fell, ${res.stuck} stuck, avg course ${avg.toFixed(0)} s` +
  Object.entries(failAt).map(([k, v]) => `\n   ${k}: ${v.join(', ')}`).join(''));
