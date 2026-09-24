// Web Worker: MuJoCo (WASM) + the Extreme Parkour teacher policy on courses generated here, in real
// time (or slower: `rate`). Posts the course grid when it changes and the Go2's body poses ~60x/s.
import loadMujoco from '../mujoco/mujoco.js';
import { TeacherPolicy } from './policy.js';
import { Go2Parkour, quatToRpy } from './sim.js';
import { TYPES, makeCourse, stuckAfter } from './course.js';

// seconds to keep running after the course ends (finish / fall / stuck / click) before the fade
const OUTRO = { done: 1.2, fell: 0.9, stuck: 0.2, next: 0 }, FADE = 0.35, DT = 0.002;

let S = null, running = false, wanted = false, last = 0;   // `wanted`: page says visible (may arrive before init finishes)

onmessage = async ({ data: m }) => {
  if (m.type === 'pause') { wanted = false; running = false; }
  else if (m.type === 'resume') { wanted = true; if (S) start(); }
  else if (m.type === 'init') {
    S = await create(m);
    sendCourse();
    if (wanted) start();
  } else if (m.type === 'next' && S && !S.end) S.end = { why: 'next', at: S.t };
  else if (m.type === 'vx' && S) S.env.vx = m.v;                  // forward speed command, 0.3-0.8 m/s
  else if (m.type === 'rate' && S) S.rate = m.r;                  // time scale (slow motion)
};

// init carries the preloaded MuJoCo binary and policy weights (fetched here otherwise)
async function create({ base, bodies, wasm, policy: weights }) {
  const [mujoco, policy, xml] = await Promise.all([
    loadMujoco(wasm ? { wasmBinary: wasm } : {}),
    weights ? fetch(base + 'policy.json').then((r) => r.json()).then((m) => new TeacherPolicy(m, new Float32Array(weights))) : TeacherPolicy.load(base),
    fetch(base + 'go2_phys.xml').then((r) => r.text()),
  ]);
  const seed = Math.floor(Math.random() * 1e6);
  const env = new Go2Parkour(mujoco, xml, policy, makeCourse(seed), { zRange: [-1.2, 1.8] });
  const ids = bodies.map((n) => mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY.value, n));
  return { env, ids, seed, poses: new Float32Array(ids.length * 7), acc: 0, t: 0, goalT: 0, lastGoal: 0, end: null, rate: 1 };
}

function sendCourse() {
  const tr = S.env.track, C = tr.C.slice();
  postMessage({ type: 'course', C, cols: tr.cols, used: tr.used, x0: tr.x0, spans: tr.spans,
    labels: tr.subs.map((s) => TYPES[s.type].label), difficulties: tr.subs.map((s) => s.difficulty) }, [C.buffer]);
}

function newCourse() {
  S.seed += 1;
  S.env.setTrack(makeCourse(S.seed));
  Object.assign(S, { t: 0, goalT: 0, lastGoal: 0, end: null });
  sendCourse();
}

function start() {
  if (running) return;
  running = true;
  last = performance.now();
  loop();
}

// after each control step: finished, fell over, or stuck without reaching a waypoint
function check() {
  const env = S.env, q = env.d.qpos;
  if (env.goalIdx !== S.lastGoal) { S.lastGoal = env.goalIdx; S.goalT = S.t; }
  if (S.end) return;
  const [r, p] = quatToRpy(q[3], q[4], q[5], q[6]);
  const why = env.finished ? 'done' : Math.abs(r) > 1.2 || Math.abs(p) > 1.2 ? 'fell' : S.t - S.goalT > stuckAfter(env.vx) ? 'stuck' : '';
  if (why) S.end = { why, at: S.t };
}

function loop() {
  if (!running) return;
  const now = performance.now();
  S.acc += Math.min((now - last) / 1000, 0.05) * S.rate;
  last = now;
  while (S.acc >= DT) {
    S.acc -= DT; S.t += DT;
    if (S.env.tick()) check();
  }
  const e = S.end;
  if (e && !e.fade && S.t - e.at >= OUTRO[e.why]) { e.fade = S.t; postMessage({ type: 'fade', why: e.why }); }
  if (e && e.fade && S.t - e.fade >= FADE) newCourse();

  const xp = S.env.d.xpos, xq = S.env.d.xquat, P = S.poses;
  S.ids.forEach((b, k) => {
    for (let j = 0; j < 3; j++) P[k * 7 + j] = xp[b * 3 + j];
    for (let j = 0; j < 4; j++) P[k * 7 + 3 + j] = xq[b * 4 + j];
  });
  const spans = S.env.track.spans, k = spans.findIndex(([, b]) => P[0] < b + 0.4), seg = k < 0 ? spans.length - 1 : k;
  postMessage({ type: 'frame', poses: P, seg, t: S.t, end: S.end?.why ?? '' });
  setTimeout(loop, 16);
}
