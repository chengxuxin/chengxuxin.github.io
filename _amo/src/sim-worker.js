// Web Worker: MuJoCo (WASM) + AMO's released policy, in real time. Receives commands and hand targets
// from the page; posts body poses, the three drag points and the measured pelvis height / torso
// orientation ~60x/s.
import loadMujoco from '../mujoco/mujoco.js';
import { AMOPolicy } from './policy.js';
import { AMOSim, DEFAULT } from './sim.js';

const FALL_Z = 0.3;
let S = null, running = false, wanted = false, last = 0;

onmessage = async ({ data: m }) => {
  if (m.type === 'init') {
    S = await create(m);
    postMessage({ type: 'ready' });
    if (wanted) start();
  } else if (m.type === 'run') {
    wanted = m.on;
    if (wanted && S) start(); else running = false;
  } else if (S && m.type === 'cmd') S.sim.commands.set(m.c);
  else if (S && m.type === 'hand') S.sim.setHandTarget(m.side, m.target);     // a hand reaches for a world point (null: hold)
  else if (S && m.type === 'rest') { S.sim.handTarget = [null, null]; S.sim.setArms(DEFAULT.slice(15)); }
  else if (S && m.type === 'reset') S.sim.reset();
};

// init carries the preloaded MuJoCo binary, policy weights and physics model (fetched here otherwise)
async function create({ base, bodies, wasm, policy: weights, mjb: mjbBytes }) {
  const [mujoco, policy, mjb] = await Promise.all([
    loadMujoco(wasm ? { wasmBinary: wasm } : {}),
    weights ? fetch(base + 'amo.json').then((r) => r.json()).then((m) => new AMOPolicy(m, weights)) : AMOPolicy.load(base),
    mjbBytes ?? fetch(base + 'g1_amo.mjb').then((r) => r.arrayBuffer()),
  ]);
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer('g1_amo.mjb', new Uint8Array(mjb));
  const model = mujoco.MjModel.from_binary_path('g1_amo.mjb', vfs);
  const sim = new AMOSim(mujoco, model, policy);
  const body = (n) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, n);
  return { sim, ids: bodies.map(body), torso: body('torso_link'), poses: new Float32Array(bodies.length * 7), acc: 0, fell: 0 };
}

function start() {
  if (running) return;
  running = true;
  last = performance.now();
  loop();
}

// torso orientation relative to the pelvis heading (what the torso commands mean): [roll, pitch, yaw]
function torsoRpy(xq, t, pw, pz) {
  const w = xq[t * 4], x = xq[t * 4 + 1], y = xq[t * 4 + 2], z = xq[t * 4 + 3];
  const hy = Math.atan2(2 * pw * pz, 1 - 2 * pz * pz);                  // pelvis yaw (small roll/pitch ignored)
  const c = Math.cos(hy / 2), s = Math.sin(hy / 2);                     // q_rel = q_heading^-1 * q_torso
  const rw = c * w + s * z, rx = c * x + s * y, ry = c * y - s * x, rz = c * z - s * w;
  return [Math.atan2(2 * (rw * rx + ry * rz), 1 - 2 * (rx * rx + ry * ry)),
    Math.asin(Math.max(-1, Math.min(1, 2 * (rw * ry - rz * rx)))),
    Math.atan2(2 * (rw * rz + rx * ry), 1 - 2 * (ry * ry + rz * rz))];
}

function loop() {
  if (!running) return;
  const now = performance.now(), sim = S.sim, d = sim.d;
  S.acc += Math.min((now - last) / 1000, 0.05);
  last = now;
  while (S.acc >= 0.002) { S.acc -= 0.002; sim.substep(); }

  const q = d.qpos;
  if (!S.fell && q[2] < FALL_Z) {                      // rare (commands far outside what it was trained on)
    S.fell = now; postMessage({ type: 'fell' });
  }
  if (S.fell && now - S.fell > 900) { sim.reset(); S.fell = 0; postMessage({ type: 'reset' }); }

  const xp = d.xpos, xq = d.xquat, P = S.poses;
  S.ids.forEach((b, k) => {
    for (let j = 0; j < 3; j++) P[k * 7 + j] = xp[b * 3 + j];
    for (let j = 0; j < 4; j++) P[k * 7 + 3 + j] = xq[b * 4 + j];
  });
  const heading = Math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
  const handles = [0, 1, 2].flatMap((h) => sim.point(h));              // chest, left hand, right hand
  postMessage({ type: 'frame', t: sim.i * 0.002, poses: P, handles, pelvis: [q[0], q[1], q[2]], heading, torso: torsoRpy(xq, S.torso, q[3], q[6]) });
  setTimeout(loop, 16);
}
