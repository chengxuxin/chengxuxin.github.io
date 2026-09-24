// Extreme Parkour terrains (port of Isaaclab_Parkour's generators, which follow arXiv 2309.14341),
// generated live in the browser. Heights are integer units of VS on a HS grid, like the original.
export const HS = 0.08, VS = 0.005, NUM_GOALS = 8;
const SLOPE_T = 1.5 * HS / VS * VS;                           // height jump (m) that Isaac turns into a vertical wall
const BASE = { platformLen: 2.5, padWidth: 0.1, downScale: 0.075, noise: [0.02, 0.06], noiseStep: 0.005 };

// ---------------------------------------------------------------- seeded RNG (mulberry32)
export function makeRng(seed) {
  let a = seed >>> 0;
  const next = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { next, uniform: (lo, hi) => lo + (hi - lo) * next(), integers: (lo, hi) => lo + Math.floor(next() * (hi - lo)) };
}

// Python's round(): half to even (1.0 / 0.08 = 12.5 -> 12 matters for the stone width)
const round = (v) => { const f = Math.floor(v), r = v - f; return r > 0.5 ? f + 1 : r < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };

// ---------------------------------------------------------------- grids
class Grid {
  constructor(w, l, fill = 0) { this.w = w; this.l = l; this.a = new Float64Array(w * l).fill(fill); }
  set(x0, x1, y0, y1, v) {             // Python-style half-open slices, negative/None handled by caller
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(this.w, x1); y1 = Math.min(this.l, y1);
    for (let i = x0; i < x1; i++) for (let j = y0; j < y1; j++) this.a[i * this.l + j] = typeof v === 'function' ? v(i - x0, j - y0) : v;
  }
  get(i, j) { return this.a[i * this.l + j]; }
}

const CFG = {
  parkour_gap: { x: [0.8, 1.5], y: [-0.4, 0.4], hvw: [0.6, 1.2], gap: (d) => 0.1 + 0.7 * d, depth: [0.2, 1.0] },
  parkour_hurdle: { x: [1.2, 2.2], y: [-0.4, 0.4], hvw: [0.4, 0.8], stone: (d) => 0.1 + 0.3 * d, hurdle: (d) => [0.1 + 0.1 * d, 0.15 + 0.25 * d] },
  parkour_step: { x: [0.3, 1.5], y: [-0.4, 0.4], hvw: [0.5, 1.0], step: (d) => 0.1 + 0.35 * d },
  parkour: { x: (d) => [-0.1, 0.1 + 0.3 * d], y: (d) => [0.2, 0.3 + 0.1 * d], pit: [0.2, 1.0], stoneW: 1.0, lastLen: 1.6,
    stone: (d) => [0.9 - 0.3 * d, 1 - 0.2 * d], incline: (d) => 0.25 * d, lastIncline: (d, ih) => ih + 0.1 - 0.1 * d },
};

function gap(d, rng, W, L) {
  const c = CFG.parkour_gap, h = new Grid(W, L), mid = L >> 1;
  const g = round(c.gap(d) / HS);
  const xmin = round(c.x[0] / HS) + g, xmax = round(c.x[1] / HS) + g, ymin = round(c.y[0] / HS), ymax = round(c.y[1] / HS);
  const pl = round(BASE.platformLen / HS);
  const depth = -round(rng.uniform(...c.depth) / VS), hvw = round(rng.uniform(...c.hvw) / HS);
  const goals = [[pl - 1, mid]];
  let x = pl, last = pl;
  for (let i = 0; i < NUM_GOALS - 2; i++) {
    const rx = rng.integers(xmin, xmax); x += rx; const ry = rng.integers(ymin, ymax);
    h.set(x - (g >> 1), x + (g >> 1), 0, L, depth);
    h.set(last, x, 0, mid + ry - hvw, depth); h.set(last, x, mid + ry + hvw, L, depth);
    last = x; goals.push([x - Math.floor(rx / 2), mid + ry]);
  }
  let fx = x + rng.integers(xmin, xmax); if (fx > W) fx = W - 6;
  goals.push([fx, mid]);
  return [h, goals];
}

function hurdle(d, rng, W, L) {
  const c = CFG.parkour_hurdle, h = new Grid(W, L), mid = L >> 1, sl = round(c.stone(d) / HS);
  const xmin = round(c.x[0] / HS), xmax = round(c.x[1] / HS), ymin = round(c.y[0] / HS), ymax = round(c.y[1] / HS);
  const hvw = round(rng.uniform(...c.hvw) / HS), [lo, hi] = c.hurdle(d), hmax = round(hi / VS), hmin = round(lo / VS);
  const pl = round(BASE.platformLen / HS);
  const goals = [[pl - 1, mid]];
  let x = pl;
  for (let i = 0; i < NUM_GOALS - 2; i++) {
    const rx = rng.integers(xmin, xmax), ry = rng.integers(ymin, ymax); x += rx;
    h.set(x - (sl >> 1), x + (sl >> 1), 0, L, rng.integers(hmin, hmax));
    h.set(x - (sl >> 1), x + (sl >> 1), 0, mid + ry - hvw, 0); h.set(x - (sl >> 1), x + (sl >> 1), mid + ry + hvw, L, 0);
    goals.push([x - Math.floor(rx / 2), mid + ry]);
  }
  let fx = x + rng.integers(xmin, xmax); if (fx > W) fx = W - 6;
  goals.push([fx, mid]);
  return [h, goals];
}

function step(d, rng, W, L) {
  const c = CFG.parkour_step, h = new Grid(W, L), mid = L >> 1, sh = round(c.step(d) / VS);
  const xmin = round(c.x[0] / HS), xmax = round(c.x[1] / HS), ymin = round(c.y[0] / HS), ymax = round(c.y[1] / HS);
  const hvw = round(rng.uniform(...c.hvw) / HS), pl = round(BASE.platformLen / HS), n = NUM_GOALS - 2;
  const goals = [[pl - round(1 / HS), mid]];
  let x = pl, last = pl, stair = 0;
  for (let i = 0; i < n; i++) {
    const rx = rng.integers(xmin, xmax), ry = rng.integers(ymin, ymax);
    if (i < (n >> 1)) stair += sh; else if (i > (n >> 1)) stair -= sh;
    h.set(x, x + rx, 0, L, stair); x += rx;
    h.set(last, x, 0, mid + ry - hvw, 0); h.set(last, x, mid + ry + hvw, L, 0);
    last = x; goals.push([x - Math.floor(rx / 2), mid + ry]);
  }
  let fx = x + rng.integers(xmin, xmax); if (fx > W) fx = W - 6;
  goals.push([fx, mid]);
  return [h, goals];
}

function stones(d, rng, W, L) {
  const c = CFG.parkour, mid = L >> 1;
  const h = new Grid(W, L, -round(rng.uniform(...c.pit) / VS));
  let sl = rng.uniform(...c.stone(d)); sl = 2 * Math.round((sl / 2) * 10) / 10; sl = round(sl / HS);
  const xr = c.x(d), yr = c.y(d);
  const xmin = sl + round(xr[0] / HS), xmax = sl + round(xr[1] / HS), ymin = round(yr[0] / HS), ymax = round(yr[1] / HS);
  const pl = round(BASE.platformLen / HS); h.set(0, pl, 0, L, 0);
  const sw = round(c.stoneW / HS), lsl = round(c.lastLen / HS);
  let ih = c.incline(d); const lih = round(c.lastIncline(d, ih) / VS); ih = round(ih / VS);
  let x = pl - rng.integers(xmin, xmax) + (sl >> 1);
  const goals = [[pl - (sl >> 1), mid]];
  let flag = rng.integers(0, 2); const n = NUM_GOALS - 2;
  const ramp = (hgt, pn) => (i, j) => Math.trunc((-hgt + (2 * hgt * j) / (sw - 1)) * pn);
  for (let i = 0; i < n; i++) {
    x += rng.integers(xmin, xmax);
    const pn = Math.round(2 * (flag - 0.5)), y = mid + pn * rng.integers(ymin, ymax);
    if (i === n - 1) { x += lsl >> 2; h.set(x - (lsl >> 1), x + (lsl >> 1), y - (sw >> 1), y + (sw >> 1), ramp(lih, pn)); }
    else h.set(x - (sl >> 1), x + (sl >> 1), y - (sw >> 1), y + (sw >> 1), ramp(ih, pn));
    goals.push([x, y]); flag = 1 - flag;
  }
  const fx = x + 2 * rng.integers(xmin, xmax);
  h.set(x + (lsl >> 1), W, 0, L, 0);
  goals.push([fx, mid]);
  return [h, goals];
}

const GEN = { parkour_gap: gap, parkour_hurdle: hurdle, parkour_step: step, parkour: stones };
export const TERRAIN_TYPES = Object.keys(GEN);

// natural cubic spline through y[] at uniform knots, evaluated at m uniform points over the same span
function resample(y, m) {
  const n = y.length, M = new Float64Array(n), c = new Float64Array(n), dd = new Float64Array(n);
  for (let i = 1; i < n - 1; i++) { c[i] = 1; dd[i] = 6 * (y[i + 1] - 2 * y[i] + y[i - 1]); }
  const cp = new Float64Array(n), dp = new Float64Array(n);            // tridiagonal (1 4 1)
  for (let i = 1; i < n - 1; i++) { const den = 4 - (i > 1 ? cp[i - 1] : 0); cp[i] = 1 / den; dp[i] = (dd[i] - (i > 1 ? dp[i - 1] : 0)) / den; }
  for (let i = n - 2; i >= 1; i--) M[i] = dp[i] - cp[i] * M[i + 1];
  const out = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const t = (k * (n - 1)) / (m - 1), i = Math.min(n - 2, Math.floor(t)), u = t - i;
    out[k] = (1 - u) * y[i] + u * y[i + 1] + ((u * u * u - u) * M[i + 1] + ((1 - u) ** 3 - (1 - u)) * M[i]) / 6;
  }
  return out;
}

function roughness(d, rng, h) {
  const size = [h.w * HS, h.l * HS], wd = Math.floor(size[0] / BASE.downScale), ld = Math.floor(size[1] / BASE.downScale);
  const maxH = (BASE.noise[1] - BASE.noise[0]) * d + BASE.noise[0];
  const lo = Math.trunc(-BASE.noise[0] / VS), hi = Math.trunc(maxH / VS);
  const down = Array.from({ length: wd }, () => Float64Array.from({ length: ld }, () => rng.integers(lo, hi + 1)));
  const rows = down.map((r) => resample(r, h.l));                          // along y
  for (let j = 0; j < h.l; j++) {
    const col = resample(Float64Array.from(rows, (r) => r[j]), h.w);       // along x
    for (let i = 0; i < h.w; i++) h.a[i * h.l + j] += Math.round(col[i]);
  }
}

// One 16 x 4 m sub-terrain: { noisy, clean } heights (201 x 51, metres) + goals (metres, relative to its centre)
export function subTerrain(type, difficulty, rng) {
  const W = 199, L = 49;                                                   // inside the 1-px border
  const [z, goals] = GEN[type](difficulty, rng, W, L);
  const pw = Math.floor(BASE.padWidth / HS);                               // pad the rim to 0
  z.set(0, W, 0, pw, 0); z.set(0, W, L - pw, L, 0); z.set(0, pw, 0, L, 0); z.set(W - pw, W, 0, L, 0);
  for (let k = 0; k < z.a.length; k++) z.a[k] = Math.round(z.a[k]);
  const clean = Float64Array.from(z.a);
  roughness(difficulty, rng, z);
  const wrap = (a) => { const g = new Float64Array(201 * 51); for (let i = 0; i < W; i++) for (let j = 0; j < L; j++) g[(i + 1) * 51 + j + 1] = a[i * L + j] * VS; return g; };
  return { type, difficulty, noisy: wrap(z.a), clean: wrap(clean), goals: goals.map(([gx, gy]) => [gx * HS - 0.5 * W * HS, gy * HS - 0.5 * L * HS]) };
}

// ---------------------------------------------------------------- track
// Sub-terrains laid end to end along +x. Raw form ({H, shape, goals, n}) = the Python reference layout
// (full 16 m blocks). Live form: each block is trimmed to its obstacle section (0.8 m before the
// first goal .. 0.8 m past the last), seams cross-faded, and the grid padded flat to `cols` columns
// so every track fits the same compiled heightfield.
const LEAD = Math.round(0.8 / HS), BLEND = 4;
export const TRACK_COLS = 801;                                   // 64 m footprint: fits all 4 types (longest of 3000 random courses: 56 m)

export class Track {
  constructor(subs, cols = TRACK_COLS) {
    this.x0 = -8;
    if (subs.H) {                                   // stored as float32: snap back to exact multiples of VS
      const H = Float64Array.from(subs.H, (v) => Math.round(v / VS) * VS);
      Object.assign(this, { n: subs.n, cols: subs.shape[0], H, C: H, goals: subs.goals });
      this.length = (this.cols - 1) * HS;
      return;
    }
    this.n = subs.length; this.subs = subs; this.cols = cols;
    this.H = new Float64Array(cols * 51); this.C = new Float64Array(cols * 51);
    this.goals = []; this.spans = [];
    let at = 0;                                     // next free column
    subs.forEach((s, k) => {
      const col = (gx) => (gx + 8) / HS;            // sub-local x (m) -> vertex column (fractional)
      const a = k === 0 ? 0 : Math.max(0, Math.floor(col(s.goals[0][0])) - LEAD);
      const b = k === subs.length - 1 ? 200 : Math.min(200, Math.ceil(col(s.goals[s.goals.length - 1][0])) + LEAD);
      if (at + (b - a) + 1 > cols) throw new Error('track too long');
      const start = at === 0 ? 0 : at + 1;         // column where this block's column a lands
      for (let i = a; i <= b; i++) {
        const dst = (start + i - a) * 51, src = i * 51;
        for (let j = 0; j < 51; j++) {
          let h = s.noisy[src + j];
          if (start > 0 && i - a < BLEND) { const t = (i - a + 1) / (BLEND + 1); h = (1 - t) * this.H[(start - 1) * 51 + j] + t * h; }
          this.H[dst + j] = h; this.C[dst + j] = s.clean[src + j];
        }
      }
      for (const [gx, gy] of s.goals) this.goals.push([this.x0 + (start + col(gx) - a) * HS, gy]);
      this.spans.push([this.x0 + start * HS, this.x0 + (start + b - a) * HS]);
      at = start + b - a;
    });
    this.used = at;                                 // last column with terrain; the rest is flat
    this.length = (cols - 1) * HS;
  }
  // terrain height at world (x, y) with Isaac's slope-threshold wall rule (grid = H or C)
  height(x, y, G = this.H) {
    let fx = (x - this.x0) / HS, fy = (y + 2) / HS;
    if (fx < 0 || fy < 0 || fx > this.cols - 1 || fy > 50) return 0;
    fx = Math.min(fx, this.cols - 1 - 1e-9); fy = Math.min(fy, 50 - 1e-9);
    const i = Math.floor(fx), j = Math.floor(fy), tx = fx - i, ty = fy - j;
    const h00 = G[i * 51 + j], h10 = G[(i + 1) * 51 + j], h01 = G[i * 51 + j + 1], h11 = G[(i + 1) * 51 + j + 1];
    const lo = Math.min(h00, h10, h01, h11), hi = Math.max(h00, h10, h01, h11);
    if (hi - lo > SLOPE_T) return lo;
    return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty;
  }
}
