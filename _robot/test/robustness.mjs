// Robustness check against the exact files the site ships (assets/robot/data):
//   1. the shuffled playlist with stands in between, several minutes, no pokes
//   2. the same, with a 10 N poke (same profile as the widget) every cooldown in a random direction
// Run: npm test   (a fall = root height < 0.42 m)
import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { GMTPolicy } from '../src/policy.js';
import { GMTController } from '../src/controller.js';
import { Playlist } from '../src/playlist.js';

const D = (p) => new URL('../../assets/robot/data/' + p, import.meta.url);
const buf = (p) => { const b = fs.readFileSync(D(p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const json = (p) => JSON.parse(fs.readFileSync(D(p)));
const POKE = { force: 10, hold: 0.25, tau: 0.25, cooldown: 2 };   // keep in sync with sim-worker.js
const MINUTES = +(process.env.MINUTES || 4);

const policy = GMTPolicy.fromBuffers(json('policy.json'), buf('policy16.bin'), buf('policy_norm.bin'));
const manifest = json('motions.json');
const bufs = Object.fromEntries(manifest.clips.map((c) => [c.name, new Float32Array(buf(`motions/${c.name}.bin`))]));
const vis = json('g1_vis.json');
const mujoco = await loadMujoco(), vfs = new mujoco.MjVFS();
vfs.addBuffer('g1.mjb', new Uint8Array(buf('g1.mjb')));
const model = mujoco.MjModel.from_binary_path('g1.mjb', vfs), data = new mujoco.MjData(model);
const torso = vis.bodies[vis.geoms.find((g) => g.body === 'torso_link').b];

function run(label, pokes, seed) {
  let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const ctl = new GMTController(mujoco, model, data, policy), list = new Playlist(manifest, bufs, rnd);
  ctl.reset(); list.start(ctl);
  let falls = 0, minZ = 9, pokeT = -9, dir = [0, 0], nPokes = 0; const played = new Set();
  const T = MINUTES * 60000;
  for (let i = 0; i < T; i++) {
    const t = i / 1000, x = data.xfrc_applied;
    if (pokes && t - pokeT >= POKE.cooldown) { pokeT = t; nPokes++; const a = rnd() * 2 * Math.PI; dir = [Math.cos(a), Math.sin(a)]; }
    const tp = t - pokeT, f = pokes ? POKE.force * (tp < POKE.hold ? 1 : Math.exp(-(tp - POKE.hold) / POKE.tau)) : 0;
    x[torso * 6] = dir[0] * f; x[torso * 6 + 1] = dir[1] * f;
    ctl.tick();
    if (i % 20 === 0) { list.update(ctl); played.add(list.current.name); }
    minZ = Math.min(minZ, data.qpos[2]);
    if (data.qpos[2] < 0.42) { falls++; console.log(`  fall at ${t.toFixed(1)}s during ${list.current.name}`); ctl.reset(); list.start(ctl); }
  }
  console.log(`[${label}] ${MINUTES} min sim, ${played.size - 1} clips seen${pokes ? `, ${nPokes} pokes` : ''}: ${falls} falls, min root z ${minZ.toFixed(3)}`);
  return falls;
}

const falls = run('playlist', false, 7) + run('playlist + pokes', true, 11);
process.exitCode = falls ? 1 : 0;
