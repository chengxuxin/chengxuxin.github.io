// Footer easter egg: a Unitree G1, drawn as line art, standing on the footer rule and
// controlled live by the GMT policy.
// Physics + policy run in a Web Worker (robot-sim.js); this file renders and handles pokes.
import * as THREE from 'three';
import { Stage, LineArtRobot, GhostRobot } from './renderer.js';

// Mirrors the worker's poke profile so the visuals track the actual force.
const POKE = { hold: 0.25, tau: 0.25, cooldown: 2 };
const pokeLevel = (t) => (t < 0 ? 0 : t < POKE.hold ? 1 : Math.exp(-(t - POKE.hold) / POKE.tau));
const SVG = 'http://www.w3.org/2000/svg';

export async function mount(root) {
  const base = new URL('./data/', import.meta.url).href;
  const canvas = root.querySelector('canvas'), box = canvas.parentElement;   // box hosts the poke overlays
  const [vis, bin] = await Promise.all([
    fetch(base + 'g1_vis.json').then((r) => r.json()),
    fetch(base + 'g1_vis.bin').then((r) => r.arrayBuffer()),
  ]);
  const stage = new Stage(canvas, { distance: 5, elevation: 0.04, floor: false, groundAtBottom: true });
  const robot = new LineArtRobot(vis, bin), ghost = new GhostRobot(vis, bin);
  stage.world.add(robot.group, ghost.group);

  // ---- worker
  const worker = new Worker(new URL('./robot-sim.js', import.meta.url), { type: 'module' });
  let frame = null, shown = false, visible = true;
  worker.onmessage = ({ data: m }) => {
    if (m.type === 'frame') {
      frame = m;
      if (!shown) { shown = true; requestAnimationFrame(() => root.classList.add('is-ready')); }
    } else if (m.type === 'fall') {
      shown = false; root.classList.remove('is-ready');
    }
  };
  worker.postMessage({ type: 'init', base });

  // Run only while on screen and the tab is visible (?robot-debug: always run, for headless testing)
  const always = /[?&]robot-debug\b/.test(location.search);
  const sync = () => worker.postMessage({ type: always || (visible && !document.hidden) ? 'resume' : 'pause' });
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync(); }).observe(root);
  document.addEventListener('visibilitychange', sync);

  // ---- poke feedback overlay: ripple at the click + arrow along the push, fading with the force
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', 'rw-arrow');
  svg.innerHTML = '<defs><marker id="rw-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
    + '<path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs><line marker-end="url(#rw-head)"/>';
  box.appendChild(svg);
  const arrow = svg.querySelector('line');
  let poke = null;   // { t0, x, y, sx, sy }: click point (css px) + push direction on screen

  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const toScreen = (x, y, z) => {                 // MuJoCo world -> canvas css px
    const v = new THREE.Vector3(x, z, -y).project(stage.camera);
    return [(v.x + 1) / 2 * canvas.clientWidth, (1 - v.y) / 2 * canvas.clientHeight];
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (!frame || (poke && performance.now() - poke.t0 < POKE.cooldown * 1000)) return;
    const r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    ndc.set((px / r.width) * 2 - 1, -(py / r.height) * 2 + 1);
    ray.setFromCamera(ndc, stage.camera);
    const [tx, ty, tz] = frame.torso;
    const view = stage.camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
    const hit = new THREE.Vector3();
    if (!ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(view, new THREE.Vector3(tx, tz, -ty)), hit)) return;
    const hx = hit.x, hy = -hit.z;                          // three world -> MuJoCo xy
    let dx = tx - hx, dy = ty - hy;                         // from the clicked point toward the torso
    if (Math.hypot(dx, dy) < 0.06) { dx = view.x; dy = -view.z; }   // center click: push away from the viewer
    const d = Math.hypot(dx, dy); dx /= d; dy /= d;
    worker.postMessage({ type: 'poke', dir: [dx, dy] });

    // push direction as seen on screen, for the arrow
    const [ax, ay] = toScreen(hx, hy, hit.y), [bx, by] = toScreen(hx + dx * 0.3, hy + dy * 0.3, hit.y);
    poke = { t0: performance.now(), x: px, y: py, sx: bx - ax, sy: by - ay };

    const ripple = document.createElement('span');
    ripple.className = 'rw-ripple';
    ripple.style.left = px + 'px'; ripple.style.top = py + 'px';
    box.appendChild(ripple);
    setTimeout(() => ripple.remove(), 800);
  });

  function drawPoke(now) {
    const k = poke ? pokeLevel((now - poke.t0) / 1000) : 0;
    stage.setInk(0.85 * k, robot.lineMat);             // contours flash orange while the force acts
    const n = poke ? Math.hypot(poke.sx, poke.sy) : 0;
    if (k < 0.02 || n < 6) { svg.style.opacity = 0; return; }
    const ux = poke.sx / n, uy = poke.sy / n, L = 34;   // arrow ends at the click, pointing along the push
    arrow.setAttribute('x1', poke.x - ux * L); arrow.setAttribute('y1', poke.y - uy * L);
    arrow.setAttribute('x2', poke.x - ux * 4); arrow.setAttribute('y2', poke.y - uy * 4);
    svg.style.opacity = k;
  }

  (function render(now) {
    if (frame) {
      robot.setPoses(frame.poses);
      ghost.setPoses(frame.ghost);
      stage.follow(frame.root[0], frame.root[1]);
    }
    drawPoke(now);
    if (visible || always) stage.render();
    requestAnimationFrame(render);
  })(performance.now());
}
