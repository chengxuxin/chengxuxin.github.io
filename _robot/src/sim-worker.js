// Web Worker: MuJoCo (WASM) + the GMT policy tracking GMT's reference clips in real time.
// Keeps the 1 kHz physics off the main thread; posts body poses to the page ~60x/s.
import loadMujoco from '../mujoco/mujoco.js';
import { GMTPolicy } from './policy.js';
import { GMTController } from './controller.js';
import { Playlist } from './playlist.js';

// Poke: full force briefly, then an exponential fade (bounded impulse). 10 N with a 2 s cooldown
// never toppled the robot when poked back-to-back in random directions for 5 x 60 s.
const POKE = { force: 10, hold: 0.25, tau: 0.25, duration: 2, cooldown: 2 };
const FALL_Z = 0.42, GHOST_LEAD = 0.04;

let S = null, running = false, wanted = true, last = 0;   // `wanted`: page says visible (may arrive before init finishes)

onmessage = async ({ data: m }) => {
  if (m.type === 'pause') { wanted = false; running = false; }
  else if (m.type === 'resume') { wanted = true; if (S) start(); }
  else if (m.type === 'init') {
    S = await create(m.base);
    postMessage({ type: 'ready' });
    if (wanted) start();
  } else if (m.type === 'poke' && S) {
    if (!S.poke || S.time - S.poke.t0 > POKE.cooldown) S.poke = { t0: S.time, dir: m.dir };
  }
};

async function create(base) {
  const manifest = await fetch(base + 'motions.json').then((r) => r.json());
  const [mujoco, mjb, policy, vis, ...clips] = await Promise.all([
    loadMujoco(),
    fetch(base + 'g1.mjb').then((r) => r.arrayBuffer()),
    GMTPolicy.load(base),
    fetch(base + 'g1_vis.json').then((r) => r.json()),
    ...manifest.clips.map((c) => fetch(`${base}motions/${c.name}.bin`).then((r) => r.arrayBuffer())),
  ]);
  const bufs = Object.fromEntries(manifest.clips.map((c, i) => [c.name, new Float32Array(clips[i])]));
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer('g1.mjb', new Uint8Array(mjb));
  const model = mujoco.MjModel.from_binary_path('g1.mjb', vfs);
  const data = new mujoco.MjData(model), ghost = new mujoco.MjData(model);
  const ctl = new GMTController(mujoco, model, data, policy);
  const s = {
    mujoco, model, data, ghost, ctl, list: new Playlist(manifest, bufs), bodies: vis.bodies,
    torso: vis.bodies[vis.geoms.find((g) => g.body === 'torso_link').b],
    poses: new Float32Array(vis.bodies.length * 7), ghostPoses: new Float32Array(vis.bodies.length * 7),
    fr: new Float64Array(36), acc: 0, time: 0, poke: null,
  };
  reset(s);
  return s;
}

function reset(s) {
  s.ctl.reset();
  s.list.start(s.ctl);
  s.acc = 0; s.time = 0; s.poke = null;
  s.data.xfrc_applied.fill(0);
}

function start() {
  if (running) return;
  running = true;
  last = performance.now();
  loop();
}

function applyPoke(s) {
  const x = s.data.xfrc_applied, o = s.torso * 6;
  x[o] = 0; x[o + 1] = 0;
  if (!s.poke) return;
  const t = s.time - s.poke.t0;
  if (t > POKE.duration) { s.poke = t > POKE.cooldown ? null : s.poke; return; }
  const f = POKE.force * (t < POKE.hold ? 1 : Math.exp(-(t - POKE.hold) / POKE.tau));
  x[o] = s.poke.dir[0] * f; x[o + 1] = s.poke.dir[1] * f;
}

function loop() {
  if (!running) return;
  const s = S, now = performance.now();
  s.acc += Math.min((now - last) / 1000, 0.05);
  last = now;
  const n = Math.floor(s.acc * 1000);
  s.acc -= n / 1000;
  for (let i = 0; i < n; i++) {
    applyPoke(s); s.ctl.tick(); s.time += 0.001;
    if (s.ctl.i % 20 === 0) s.list.update(s.ctl);
  }

  if (s.data.qpos[2] < FALL_Z) {                  // rare: fade out, reset, fade back in
    running = false;
    postMessage({ type: 'fall' });
    setTimeout(() => { reset(s); postMessage({ type: 'ready' }); if (wanted) start(); }, 450);
    return;
  }
  readPoses(s);
  readGhost(s);
  const q = s.data.qpos, xp = s.data.xpos;
  postMessage({
    type: 'frame', poses: s.poses, ghost: s.ghostPoses,
    root: [q[0], q[1]], torso: [xp[s.torso * 3], xp[s.torso * 3 + 1], xp[s.torso * 3 + 2]],
  });
  setTimeout(loop, 16);
}

function readPoses(s) {
  const xp = s.data.xpos, xq = s.data.xquat, P = s.poses;
  s.bodies.forEach((b, k) => {
    for (let j = 0; j < 3; j++) P[k * 7 + j] = xp[b * 3 + j];
    for (let j = 0; j < 4; j++) P[k * 7 + 3 + j] = xq[b * 4 + j];
  });
}

// Reference pose the policy is tracking (slightly ahead), aligned to the robot's root xy + heading.
function readGhost(s) {
  const fr = s.ctl.motion.sample(s.ctl.step * 0.02 + GHOST_LEAD, s.fr), q = s.data.qpos, g = s.ghost.qpos;
  const yaw = (w, x, y, z) => Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const [rw, rx, ry, rz] = [fr[6], fr[3], fr[4], fr[5]];               // reference quat is xyzw
  const d = yaw(q[3], q[4], q[5], q[6]) - yaw(rw, rx, ry, rz), cw = Math.cos(d / 2), sw = Math.sin(d / 2);
  g[0] = q[0]; g[1] = q[1]; g[2] = fr[2];
  g[3] = cw * rw - sw * rz; g[4] = cw * rx - sw * ry; g[5] = cw * ry + sw * rx; g[6] = cw * rz + sw * rw;
  for (let k = 0; k < 23; k++) g[7 + k] = fr[13 + k];
  s.mujoco.mj_kinematics(s.model, s.ghost);
  const xp = s.ghost.xpos, xq = s.ghost.xquat, P = s.ghostPoses;
  s.bodies.forEach((b, k) => {
    for (let j = 0; j < 3; j++) P[k * 7 + j] = xp[b * 3 + j];
    for (let j = 0; j < 4; j++) P[k * 7 + 3 + j] = xq[b * 4 + j];
  });
}
