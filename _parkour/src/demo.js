// Extreme Parkour live demo, opened from the Highlights card. A Go2 runs the Isaaclab_Parkour teacher
// policy on courses of the paper's terrain types generated on the fly; physics, policy and course
// generation run in a Web Worker (parkour-sim.js), this file renders and wires the controls:
// new terrain, slow motion, the forward-speed command; drag the floor to look around.
import { Stage, LineArtRobot, TerrainView } from './view.js';
import { PosePlayback } from './playback.js';
import { Vector3 } from 'three';

let demo = null, files = null;                      // demo: created on first open, kept for re-opens

// Start the big downloads (hovering "try it live" calls this) without starting anything: MuJoCo,
// the policy and the meshes are kept in memory and handed to the worker on open.
export function preload() {
  if (files) return files;
  const base = new URL('./data/', import.meta.url).href, bytes = (u) => fetch(u).then((r) => r.arrayBuffer());
  files = {
    vis: fetch(base + 'go2_vis.json').then((r) => r.json()), visBin: bytes(base + 'go2_vis.bin'),
    wasm: bytes(new URL('../mujoco/mujoco.wasm', import.meta.url)), policy: bytes(base + 'policy.bin'),
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
  const course = root.querySelector('.ld-course'), time = root.querySelector('.ld-time');
  const f = preload();
  const [vis, bin] = await Promise.all([f.vis, f.visBin]);
  onProgress?.(0.2);
  const stage = new Stage(canvas), robot = new LineArtRobot(vis, bin), terrain = new TerrainView();
  stage.world.add(terrain.group, robot.group);
  const az0 = stage.azimuth, el0 = stage.elevation;

  // ---- worker
  const worker = new Worker(new URL('./parkour-sim.js', import.meta.url), { type: 'module' });
  let frame = null, seg = -1, snap = true, visible = false, ending = '', rate = 1;
  const play = new PosePlayback(vis.bodies.length);          // smooth motion at any display refresh rate
  worker.onmessage = ({ data: m }) => {
    if (m.type === 'course') {
      terrain.build(m);
      course.innerHTML = m.labels.map((l, i) => `<span>${l} <em>${m.difficulties[i].toFixed(2)}</em></span>`).join('<i>&rarr;</i>');
      seg = -1; snap = true; ending = '';
      play.reset();                                 // no old-course poses once the new terrain is in
      root.classList.remove('is-swapping');
    } else if (m.type === 'frame') {
      if (!frame) { onProgress?.(1); requestAnimationFrame(() => root.classList.add('is-live')); }
      frame = m;
      play.push(m.t, m.poses);
      if (m.seg !== seg) { seg = m.seg; [...course.querySelectorAll('span')].forEach((s, i) => s.classList.toggle('on', i === seg)); }
      if (m.end) ending = m.end;
    } else if (m.type === 'fade') root.classList.add('is-swapping');
  };
  const [wasm, policy] = await Promise.all([f.wasm, f.policy]);
  worker.postMessage({ type: 'init', base, bodies: vis.bodies, wasm, policy }, [wasm, policy]);

  // ---- controls
  const bar = root.querySelector('.ld-controls');
  const chip = (label, fn) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = label; b.onclick = fn; bar.appendChild(b); return b; };
  chip('new terrain', () => worker.postMessage({ type: 'next' }));
  const slow = chip('slow-mo', () => { const on = !slow.classList.contains('on'); slow.classList.toggle('on', on); rate = on ? 0.3 : 1; worker.postMessage({ type: 'rate', r: rate }); });
  const speed = document.createElement('label');
  speed.className = 'ld-speed';
  speed.title = 'Forward speed command given to the policy (its training range)';
  speed.innerHTML = '<span>speed</span><input type="range" min="0.3" max="0.8" step="0.05" value="0.8" aria-label="Forward speed command"><output>0.80 m/s</output>';
  bar.appendChild(speed);
  const range = speed.querySelector('input'), out = speed.querySelector('output');
  range.oninput = () => { out.textContent = `${(+range.value).toFixed(2)} m/s`; worker.postMessage({ type: 'vx', v: +range.value }); };

  // ---- drag the floor to look around (springs back slowly while the robot runs)
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { canvas.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, az: stage.azimuth, el: stage.elevation }; root.classList.add('is-touched'); });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    stage.azimuth = drag.az - (e.clientX - drag.x) * 0.006;
    stage.elevation = Math.min(1.1, Math.max(0.08, drag.el + (e.clientY - drag.y) * 0.004));
  });
  const release = () => { drag = null; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  // ---- loop
  // ?robot-debug: record where the body and a calf land on screen each frame (for checking smoothness)
  const debug = /[?&]robot-debug\b/.test(location.search) ? (window.__pkFrames = []) : null;
  let last = performance.now();
  (function render(now) {
    const dt = Math.min(0.1, ((now ?? last) - last) / 1000); last = now ?? last;
    const shown = visible && play.sample(dt, rate);
    if (shown) {
      const p = shown.poses;
      robot.setPoses(p);
      stage.follow(p[0], p[1], p[2], 0.06, snap); snap = false;
      terrain.follow(p[0], p[1]);
      if (!drag) { stage.azimuth += (az0 - stage.azimuth) * 0.004; stage.elevation += (el0 - stage.elevation) * 0.004; }
      time.textContent = ending === 'done' ? 'course cleared · new terrain' : ending === 'fell' ? 'fell · new terrain'
        : ending === 'stuck' ? 'stuck · new terrain' : `${frame.t.toFixed(1)} s`;
      stage.render();
      if (debug) debug.push([now, ...[0, 3].map((b) => { const v = robot.bodies[b].getWorldPosition(new Vector3()).project(stage.camera); return [v.x, v.y]; }).flat()]);
    }
    requestAnimationFrame(render);
  })();

  let open = false;
  const run = () => worker.postMessage({ type: open && !document.hidden ? 'resume' : 'pause' });
  document.addEventListener('visibilitychange', run);
  return {
    resume() { open = visible = true; run(); panel.focus({ preventScroll: true }); },
    pause() { open = visible = false; run(); },
  };
}
