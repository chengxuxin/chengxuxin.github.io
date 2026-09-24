// Worker-side cost of both demos: memory (WASM heap + JS) and CPU per 20 ms control step.
import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
const MB = (b) => (b / 1e6).toFixed(1) + ' MB';
const mem = () => { global.gc?.(); const m = process.memoryUsage(); return m; };
const m0 = mem();
const mujoco = await loadMujoco();
let view = null; const heap = () => view?.buffer.byteLength ?? 0;          // any model array view sits on the WASM memory
const m1 = mem();
console.log(`MuJoCo runtime loaded: process +${MB(m1.rss - m0.rss)} rss`);

// AMO
{
  const { AMOPolicy } = await import('../src/policy.js'); const { AMOSim } = await import('../src/sim.js');
  const A = (p) => new URL('../../assets/amo/data/' + p, import.meta.url), buf = fs.readFileSync(A('amo.bin'));
  const a0 = mem(), h0 = heap();
  const policy = new AMOPolicy(JSON.parse(fs.readFileSync(A('amo.json'))), buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const vfs = new mujoco.MjVFS(); vfs.addBuffer('g1_amo.mjb', new Uint8Array(fs.readFileSync(A('g1_amo.mjb'))));
  const sim = new AMOSim(mujoco, mujoco.MjModel.from_binary_path('g1_amo.mjb', vfs), policy);
  view = sim.d.qpos; const a1 = mem();
  for (let k = 0; k < 50; k++) sim.step();
  let t = performance.now(), n = 250; for (let k = 0; k < n; k++) sim.step(); const ms = (performance.now() - t) / n;
  sim.setHandTarget(0, sim.point(1).map((v, i) => v + [0.2, 0, 0.3][i])); sim.setHandTarget(1, sim.point(2).map((v, i) => v + [0.2, 0, 0.3][i]));
  t = performance.now(); for (let k = 0; k < n; k++) sim.step(); const msIK = (performance.now() - t) / n;
  t = performance.now(); for (let k = 0; k < n; k++) policy.act(sim.obs, sim.extraIn); const msPol = (performance.now() - t) / n;
  console.log(`AMO: wasm heap now ${MB(heap())}, JS arrays (weights) ~${MB(a1.arrayBuffers - a0.arrayBuffers)}, heap +${MB(a1.heapUsed - a0.heapUsed)} | per 20 ms control step: ${ms.toFixed(2)} ms (policy ${msPol.toFixed(2)} ms), ${msIK.toFixed(2)} ms with both hands reaching -> ${(ms / 20 * 100).toFixed(0)}-${(msIK / 20 * 100).toFixed(0)}% of one core`);
}
// Parkour
{
  const { TeacherPolicy } = await import('../../_parkour/src/policy.js'); const { Go2Parkour } = await import('../../_parkour/src/sim.js'); const { makeCourse } = await import('../../_parkour/src/course.js');
  const A = (p) => new URL('../../assets/parkour/data/' + p, import.meta.url), buf = fs.readFileSync(A('policy.bin'));
  const p0 = mem(), h0 = heap();
  const policy = new TeacherPolicy(JSON.parse(fs.readFileSync(A('policy.json'))), new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
  const env = new Go2Parkour(mujoco, fs.readFileSync(A('go2_phys.xml'), 'utf8'), policy, makeCourse(1), { zRange: [-1.2, 1.8] });
  const p1 = mem();
  for (let k = 0; k < 50; k++) env.step();
  let t = performance.now(), n = 500; for (let k = 0; k < n; k++) env.step(); const ms = (performance.now() - t) / n;
  t = performance.now(); env.setTrack(makeCourse(2)); const swap = performance.now() - t;
  console.log(`Parkour (same MuJoCo instance): wasm heap now ${MB(heap())}, JS arrays ~${MB(p1.arrayBuffers - p0.arrayBuffers)}, heap +${MB(p1.heapUsed - p0.heapUsed)} | per 20 ms control step: ${ms.toFixed(2)} ms -> ${(ms / 20 * 100).toFixed(0)}% of one core | new course (generate + heightfield): ${swap.toFixed(0)} ms`);
}
console.log(`total process rss ${MB(process.memoryUsage().rss)}; wasm heap ${MB(heap())}`);
