// AMO live demo, opened from the Highlights card. The released AMO policy runs in a Web Worker
// (amo-sim.js); this file renders the G1 and turns pointer / keyboard input into AMO's commands.
// Three orange drag points sit on the line art:
//   a hand          -> that hand reaches for the cursor (arm IK); the policy balances the rest
//   the chest       -> whole body: torso height (down / up), pitch (toward its front), roll (sideways)
//   scroll on robot -> torso yaw        WASD -> walk / turn        drag the floor -> orbit the camera
import { Stage, LineArtRobot, Floor } from './view.js';
import { PosePlayback } from './playback.js';

// AMO commands: [vx, target yaw, vy, height offset (+0.75 m), torso yaw, pitch, roll, arms]
const RANGE = { height: [-0.42, 0.02], yaw: [-1.4, 1.4], pitch: [-0.45, 1.5], roll: [-0.6, 0.6] };
const IDX = { height: 3, yaw: 4, pitch: 5, roll: 6 };
const PRESETS = {
  stand: { height: 0, yaw: 0, pitch: 0, roll: 0 },
  squat: { height: -0.4, yaw: 0, pitch: 0.15, roll: 0 },
  twist: { height: -0.1, yaw: 1.2, pitch: 0.2, roll: 0 },
  lean: { height: -0.05, yaw: 0, pitch: 0, roll: 0.5 },
};
const clamp = (v, [lo, hi]) => Math.min(Math.max(v, lo), hi);
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

let demo = null, files = null;                      // demo: created on first open, kept for re-opens

// Start the big downloads (hovering "try it live" calls this) without starting anything: MuJoCo,
// the policy, the physics model and the meshes are kept in memory and handed to the worker on open.
export function preload() {
  if (files) return files;
  const base = new URL('./data/', import.meta.url).href, bytes = (u) => fetch(u).then((r) => r.arrayBuffer());
  files = {
    vis: fetch(base + 'g1_vis.json').then((r) => r.json()), visBin: bytes(base + 'g1_vis.bin'),
    wasm: bytes(new URL('../mujoco/mujoco.wasm', import.meta.url)), policy: bytes(base + 'amo.bin'), mjb: bytes(base + 'g1_amo.mjb'),
  };
  return files;
}

export async function open(root, onProgress) {
  if (!demo) demo = await create(root, onProgress);
  demo.resume();
  return demo;
}

export function close() { demo?.pause(); }

async function create(root, onProgress) {
  const base = new URL('./data/', import.meta.url).href;
  const canvas = root.querySelector('.ld-canvas'), panel = root.querySelector('.ld-panel');
  const f = preload();
  const [vis, bin] = await Promise.all([f.vis, f.visBin]);
  onProgress?.(0.2);
  const stage = new Stage(canvas), robot = new LineArtRobot(vis, bin), floor = new Floor();
  stage.world.add(floor.group, robot.group);

  const worker = new Worker(new URL('./amo-sim.js', import.meta.url), { type: 'module' });
  const cmd = new Float32Array(8);
  let frame = null, ready = false, visible = false, dirty = true, anim = null, intro = true;
  const play = new PosePlayback(vis.bodies.length, ['handles', 'pelvis']);   // smooth at any refresh rate
  let shown = null;                                   // what is on screen: interpolated poses, drag points, pelvis
  const send = () => { dirty = true; };
  worker.onmessage = ({ data: m }) => {
    if (m.type === 'ready') { ready = true; onProgress?.(1); }
    else if (m.type === 'frame') {
      if (!frame) requestAnimationFrame(() => root.classList.add('is-live'));
      frame = m;
      play.push(m.t, m.poses, { handles: m.handles, pelvis: m.pelvis });
    } else if (m.type === 'fell') root.classList.add('is-fallen');
    else if (m.type === 'reset') { root.classList.remove('is-fallen'); play.reset(); cmd.fill(0); syncUI(); send(); }
  };
  const [wasm, policy, mjb] = await Promise.all([f.wasm, f.policy, f.mjb]);
  worker.postMessage({ type: 'init', base, bodies: vis.bodies, wasm, policy, mjb }, [wasm, policy, mjb]);

  // ---- gauges: command (orange) vs measured (ink); click / drag a track to set the command
  const gauges = root.querySelector('.ld-gauges'), G = {};
  for (const name of ['height', 'pitch', 'roll', 'yaw']) {
    const row = document.createElement('div');
    row.className = 'ld-gauge';
    row.innerHTML = `<span class="ld-g-name">${name}</span><span class="ld-g-track"><i class="ld-g-cmd"></i><i class="ld-g-real"></i></span><span class="ld-g-val"></span>`;
    gauges.appendChild(row);
    const track = row.querySelector('.ld-g-track');
    const setFrom = (e) => {
      const r = track.getBoundingClientRect(), [lo, hi] = RANGE[name];
      cmd[IDX[name]] = lo + Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * (hi - lo);
      stopAnim(); syncUI(); send();
    };
    track.addEventListener('pointerdown', (e) => { track.setPointerCapture(e.pointerId); setFrom(e); track.onpointermove = setFrom; });
    track.addEventListener('pointerup', () => { track.onpointermove = null; });
    G[name] = { cmd: row.querySelector('.ld-g-cmd'), real: row.querySelector('.ld-g-real'), val: row.querySelector('.ld-g-val') };
  }
  const pos = (name, v) => { const [lo, hi] = RANGE[name]; return `${Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100))}%`; };
  const fmt = (name, v) => (name === 'height' ? `${(0.75 + v).toFixed(2)} m` : `${Math.round((v * 180) / Math.PI)}°`);
  function syncUI() { for (const n in G) { G[n].cmd.style.left = pos(n, cmd[IDX[n]]); G[n].val.textContent = fmt(n, cmd[IDX[n]]); } }
  function showMeasured() {
    if (!frame) return;
    const [r, p, y] = frame.torso, real = { height: frame.pelvis[2] - 0.75, pitch: p, roll: r, yaw: y };
    for (const n in G) G[n].real.style.left = pos(n, real[n]);
  }
  syncUI();

  // ---- presets
  const bar = root.querySelector('.ld-presets');
  const chip = (label, fn) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = label; b.onclick = fn; bar.appendChild(b); return b; };
  for (const name in PRESETS) chip(name, () => { animateTo(PRESETS[name]); if (name === 'stand') worker.postMessage({ type: 'rest' }); });

  function animateTo(target, dur = 0.9, then) {
    const from = {}; for (const n in target) from[n] = cmd[IDX[n]];
    anim = { from, target, t0: performance.now(), dur, then };
  }
  function stopAnim() { anim = null; intro = false; }
  function stepAnim(now) {
    if (!anim) return;
    const k = Math.min(1, (now - anim.t0) / 1000 / anim.dur), e = ease(k);
    for (const n in anim.target) cmd[IDX[n]] = anim.from[n] + (anim.target[n] - anim.from[n]) * e;
    syncUI(); send();
    if (k >= 1) { const t = anim.then; anim = null; t?.(); }
  }

  // ---- pointer: the three drag points (chest, left hand, right hand), the rest of the robot (= chest),
  // or the floor (orbit)
  const svg = root.querySelector('.ld-tether'), tether = svg.querySelector('line');
  const dots = [0, 1, 2].map((h) => {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'ld-h');
    g.innerHTML = '<circle class="ld-h-ring" r="7"/><circle class="ld-h-dot" r="6"/>';
    svg.appendChild(g); return g;
  });
  const HIT = matchMedia('(pointer: coarse)').matches ? 28 : 18;      // px; bigger for fingers
  let drag = null, hover = null, handTarget = null;
  const local = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const handleXY = (h) => stage.toScreen(shown.handles[h * 3], shown.handles[h * 3 + 1], shown.handles[h * 3 + 2]);
  function hit(x, y) {                                // nearest drag point, else any part of the robot = chest
    if (!shown) return null;
    let best = null, bd = HIT;
    for (let h = 0; h < 3; h++) { const [hx, hy] = handleXY(h), d = Math.hypot(hx - x, hy - y); if (d < bd) { bd = d; best = h; } }
    return best ?? (stage.pick(x, y) >= 0 ? 0 : null);
  }
  canvas.addEventListener('pointermove', (e) => {
    const [x, y] = local(e);
    if (drag?.kind === 'grab') {
      const dx = x - drag.x, dy = y - drag.y, { right, up, mpp } = drag.basis;
      const w = [0, 1, 2].map((i) => (right[i] * dx - up[i] * dy) * mpp);           // cursor offset in metres (world)
      if (drag.h === 0) {
        const c = Math.cos(drag.heading), s = Math.sin(drag.heading);
        const fwd = w[0] * c + w[1] * s, left = -w[0] * s + w[1] * c;
        cmd[3] = clamp(drag.c[3] + w[2], RANGE.height);
        cmd[5] = clamp(drag.c[5] + fwd * 3.2, RANGE.pitch);
        cmd[6] = clamp(drag.c[6] - left * 2.4, RANGE.roll);
        syncUI(); send();
      } else handTarget = drag.p0.map((v, i) => v + w[i]);                          // sent once per frame
      tether.setAttribute('x2', x); tether.setAttribute('y2', y);
    } else if (drag?.kind === 'orbit') {
      stage.azimuth = drag.az - (x - drag.x) * 0.008;
      stage.elevation = Math.min(0.9, Math.max(-0.05, drag.el + (y - drag.y) * 0.004));
    } else {
      hover = hit(x, y);
      canvas.style.cursor = hover !== null ? 'grab' : 'default';
    }
  });
  canvas.addEventListener('pointerdown', (e) => {
    const [x, y] = local(e);
    canvas.setPointerCapture(e.pointerId);
    const h = hit(x, y);
    if (h !== null) {
      if (h === 0) stopAnim();
      const p0 = shown.handles.slice(h * 3, h * 3 + 3);
      drag = { kind: 'grab', h, x, y, p0, c: Float32Array.from(cmd), basis: stage.screenBasis(h ? p0 : null), heading: frame.heading };
      tether.setAttribute('x2', x); tether.setAttribute('y2', y);
      root.classList.add('is-grabbing'); canvas.style.cursor = 'grabbing';
    } else drag = { kind: 'orbit', x, y, az: stage.azimuth, el: stage.elevation };
    root.classList.add('is-touched');
  });
  const release = () => {
    if (drag?.kind === 'grab' && drag.h) {
      if (handTarget) worker.postMessage({ type: 'hand', side: drag.h - 1, target: handTarget });
      handTarget = null; worker.postMessage({ type: 'hand', side: drag.h - 1, target: null });
    }
    drag = null; root.classList.remove('is-grabbing'); canvas.style.cursor = hover !== null ? 'grab' : 'default';
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('wheel', (e) => {
    if (hover === null && !drag) return;             // only over the robot, so the page can still scroll
    e.preventDefault();
    stopAnim();
    cmd[4] = clamp(cmd[4] - e.deltaY * 0.0025, RANGE.yaw);
    syncUI(); send(); root.classList.add('is-touched');
  }, { passive: false });

  // ---- keyboard: WASD walk / turn (turning needs forward speed, as in AMO's own play script)
  const keys = new Set();
  const onKey = (e) => {
    const k = e.key.toLowerCase();
    if (!'wasdqe'.includes(k) || k.length !== 1 || e.metaKey || e.ctrlKey) return;
    if (e.type === 'keydown') keys.add(k); else keys.delete(k);
    e.preventDefault(); root.classList.add('is-touched');
  };
  root.addEventListener('keydown', onKey);
  root.addEventListener('keyup', onKey);
  function walk(dt) {
    const vx = (keys.has('w') ? 0.4 : 0) - (keys.has('s') ? 0.3 : 0), vy = (keys.has('q') ? 0.3 : 0) - (keys.has('e') ? 0.3 : 0);
    const turn = (keys.has('a') ? 0.8 : 0) - (keys.has('d') ? 0.8 : 0);
    const k = Math.min(1, dt * 4);
    const nvx = cmd[0] + (vx - cmd[0]) * k, nvy = cmd[2] + (vy - cmd[2]) * k;
    const snap = (v) => (Math.abs(v) < 0.02 ? 0 : v);
    if (snap(nvx) !== cmd[0] || snap(nvy) !== cmd[2] || turn) {
      cmd[0] = snap(nvx); cmd[2] = snap(nvy);
      if (turn && frame) cmd[1] = frame.heading + turn * 0.6;                 // aim a little ahead of the current heading
      send();
    }
    if (!turn && frame && Math.abs(cmd[0]) < 0.1) cmd[1] = frame.heading;
  }

  // ---- loop
  let lastT = performance.now();
  function render(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
    if (visible) {
      stepAnim(now); walk(dt);
      if (dirty && ready) { worker.postMessage({ type: 'cmd', c: cmd }); dirty = false; }
      shown = play.sample(dt) ?? shown;
      if (shown) {
        robot.setPoses(shown.poses);
        stage.follow(shown.pelvis[0], shown.pelvis[1]);
        floor.follow(shown.pelvis[0], shown.pelvis[1]);
        showMeasured();
        for (let h = 0; h < 3; h++) {
          const [hx, hy] = handleXY(h), active = drag?.kind === 'grab' && drag.h === h;
          dots[h].setAttribute('transform', `translate(${hx},${hy})`);
          dots[h].classList.toggle('is-hot', active || (!drag && hover === h));
          dots[h].classList.toggle('is-active', active);
          if (active) { tether.setAttribute('x1', hx); tether.setAttribute('y1', hy); }
        }
        if (handTarget && drag) { worker.postMessage({ type: 'hand', side: drag.h - 1, target: handTarget }); handTarget = null; }
      }
      stage.setRobotTint(drag?.kind === 'grab' ? 0.8 : hover !== null ? 0.4 : 0);
      stage.render();
    }
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);

  // a short bow once it is live, so the robot reads as alive; any input cancels it
  const introTimer = setInterval(() => {
    if (!frame || !intro) return;
    clearInterval(introTimer);
    animateTo({ pitch: 0.75, height: -0.08 }, 1.0, () => setTimeout(() => intro && animateTo({ pitch: 0, height: 0 }, 1.0), 500));
  }, 200);

  let open = false;
  const run = () => worker.postMessage({ type: 'run', on: open && !document.hidden });
  document.addEventListener('visibilitychange', run);
  return {
    resume() { open = visible = true; run(); lastT = performance.now(); panel.focus({ preventScroll: true }); },
    pause() { open = visible = false; keys.clear(); run(); },
  };
}
