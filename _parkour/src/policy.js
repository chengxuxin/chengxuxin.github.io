// Extreme Parkour teacher (Isaaclab_Parkour Go2), play-time path: estimator -> priv_explicit,
// history encoder -> latent (hist_encoding=True). Plain JS forward pass.
const elu = (x) => (x > 0 ? x : Math.expm1(x));

function linear(W, b, x, out, nIn, nOut, act) {
  for (let o = 0; o < nOut; o++) {
    let s = b[o]; const r = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[r + i] * x[i];
    out[o] = act ? act(s) : s;
  }
  return out;
}

export class TeacherPolicy {
  constructor(manifest, weights) {
    this.p = {};
    for (const e of manifest.params) {
      const n = e.shape.reduce((a, b) => a * b, 1);
      this.p[e.name] = weights.subarray(e.offset, e.offset + n);
    }
    const P = (k) => [this.p[k + '.weight'], this.p[k + '.bias']];
    this.L = P;
    this.buf = { e1: new Float64Array(128), e2: new Float64Array(64), est: new Float64Array(9),
      s1: new Float64Array(128), s2: new Float64Array(64), scan: new Float64Array(32),
      proj: new Float64Array(10 * 30), c1: new Float64Array(20 * 4), c2: new Float64Array(10 * 3), hist: new Float64Array(20),
      x: new Float64Array(114), h1: new Float64Array(512), h2: new Float64Array(256), h3: new Float64Array(128), a: new Float64Array(12) };
  }

  static async load(base) {
    const [m, w] = await Promise.all([fetch(base + 'policy.json').then((r) => r.json()), fetch(base + 'policy.bin').then((r) => r.arrayBuffer())]);
    return new TeacherPolicy(m, new Float32Array(w));
  }

  // prop: 53, scan: 132, hist: 10 x 53 (row-major, oldest first) -> 12 actions (Isaac joint order)
  act(prop, scan, hist) {
    const B = this.buf, L = this.L;
    linear(...L('est:estimator.0'), prop, B.e1, 53, 128, elu);
    linear(...L('est:estimator.2'), B.e1, B.e2, 128, 64, elu);
    linear(...L('est:estimator.4'), B.e2, B.est, 64, 9);

    linear(...L('actor:actor.scan_encoder.0'), scan, B.s1, 132, 128, elu);
    linear(...L('actor:actor.scan_encoder.2'), B.s1, B.s2, 128, 64, elu);
    linear(...L('actor:actor.scan_encoder.4'), B.s2, B.scan, 64, 32, Math.tanh);

    // history encoder: per-step Linear(53->30)+ELU, Conv1d(30->20,k4,s2)+ELU, Conv1d(20->10,k2,s1)+ELU, Linear(30->20)+ELU
    const [ew, eb] = L('actor:actor.history_encoder.encoder.0'), row = new Float64Array(30);
    for (let t = 0; t < 10; t++) {
      linear(ew, eb, hist.subarray(t * 53, t * 53 + 53), row, 53, 30, elu);
      for (let c = 0; c < 30; c++) B.proj[c * 10 + t] = row[c];     // channel-major [30][10]
    }
    const conv = (W, b, x, out, cin, len, cout, k, stride) => {
      const lo = Math.floor((len - k) / stride) + 1;
      for (let c = 0; c < cout; c++) for (let t = 0; t < lo; t++) {
        let s = b[c];
        for (let ci = 0; ci < cin; ci++) for (let j = 0; j < k; j++) s += W[(c * cin + ci) * k + j] * x[ci * len + t * stride + j];
        out[c * lo + t] = elu(s);
      }
      return lo;
    };
    const l1 = conv(...L('actor:actor.history_encoder.conv_layers.0'), B.proj, B.c1, 30, 10, 20, 4, 2);
    conv(...L('actor:actor.history_encoder.conv_layers.2'), B.c1, B.c2, 20, l1, 10, 2, 1);
    linear(...L('actor:actor.history_encoder.linear_output.0'), B.c2, B.hist, 30, 20, elu);

    const x = B.x;
    x.set(prop, 0); x.set(B.scan, 53); x.set(B.est, 85); x.set(B.hist, 94);
    linear(...L('actor:actor.actor_backbone.0'), x, B.h1, 114, 512, elu);
    linear(...L('actor:actor.actor_backbone.2'), B.h1, B.h2, 512, 256, elu);
    linear(...L('actor:actor.actor_backbone.4'), B.h2, B.h3, 256, 128, elu);
    return linear(...L('actor:actor.actor_backbone.6'), B.h3, B.a, 128, 12);
  }
}
