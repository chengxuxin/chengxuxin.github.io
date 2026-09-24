// AMO (RSS 2025) released student policy + adapter, forward pass in plain JS.
// Mirrors amo_jit.pt's TorchScript forward (see py/amo_np.py). Big layers ship as int8 per output
// row: out = scale * sum(q * x) + b.
const elu = (x) => (x > 0 ? x : Math.expm1(x));
const lrelu = (x) => (x > 0 ? x : 0.01 * x);

function f16(h) {
  const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

// W: {q, s} (int8 + row scales) or Float32Array; x may be any array-like
function linear(W, b, x, xo, out, nIn, nOut, act) {
  if (W.q) {
    const { q, s } = W;
    for (let o = 0; o < nOut; o++) {
      let acc = 0; const r = o * nIn;
      for (let i = 0; i < nIn; i++) acc += q[r + i] * x[xo + i];
      const v = acc * s[o] + b[o]; out[o] = act ? act(v) : v;
    }
  } else {
    for (let o = 0; o < nOut; o++) {
      let acc = b[o]; const r = o * nIn;
      for (let i = 0; i < nIn; i++) acc += W[r + i] * x[xo + i];
      out[o] = act ? act(acc) : acc;
    }
  }
  return out;
}

export class AMOPolicy {
  constructor(manifest, buf) {
    this.p = {};
    for (const e of manifest.params) {
      const n = e.shape.reduce((a, b) => a * b, 1);
      if (e.kind === 'f32') this.p[e.name] = new Float32Array(buf, e.offset, n);
      else if (e.kind === 'f16') this.p[e.name] = Float32Array.from(new Uint16Array(buf, e.offset, n), f16);
      else this.p[e.name] = { q: new Int8Array(buf, e.offset, n), s: new Float32Array(buf, e.offset + n + ((4 - (n % 4)) % 4), e.shape[0]) };
    }
    this.norm = manifest.norm;
    const B = (n) => new Float64Array(n);
    this.b = { t1: B(128), t2: B(16), tcat: B(64), text: B(16), row: B(30), proj: B(300), c1: B(80), c2: B(30), hl: B(20),
      x: B(2474), h1: B(1024), h2: B(1024), h3: B(512), a: B(15), ax: B(12), a1: B(512), a2: B(512), a3: B(256), ay: B(15) };
  }

  static async load(base) {
    const [m, w] = await Promise.all([fetch(base + 'amo.json').then((r) => r.json()), fetch(base + 'amo.bin').then((r) => r.arrayBuffer())]);
    return new AMOPolicy(m, w);
  }

  L(k) { return [this.p[k + '.weight'], this.p[k + '.bias']]; }

  // [height, torso yaw, pitch, roll, 8 arm dof] -> 15 lower-body references
  adapter(x12) {
    const { b, norm: n } = this, P = this.p;
    for (let i = 0; i < 12; i++) b.ax[i] = (x12[i] - n.input_mean[i]) / (n.input_std[i] + 1e-8);
    let x = b.ax, nIn = 12;
    for (const [lin, bn, out, nOut] of [[0, 1, b.a1, 512], [3, 4, b.a2, 512], [6, 7, b.a3, 256]]) {
      linear(...this.L(`adapter.model.${lin}`), x, 0, out, nIn, nOut);
      const g = P[`adapter.model.${bn}.weight`], be = P[`adapter.model.${bn}.bias`], mu = P[`adapter.model.${bn}.running_mean`], va = P[`adapter.model.${bn}.running_var`];
      for (let i = 0; i < nOut; i++) out[i] = lrelu((out[i] - mu[i]) / Math.sqrt(va[i] + 1e-5) * g[i] + be[i]);
      x = out; nIn = nOut;
    }
    linear(...this.L('adapter.model.9'), x, 0, b.ay, 256, 15);
    for (let i = 0; i < 15; i++) b.ay[i] = b.ay[i] * n.output_std[i] + n.output_mean[i];
    return b.ay;
  }

  // obs: 1043 = [prop 93 | demo 17 | priv 3 | hist 10 x 93]; extra: 25 x 93 -> 15 raw actions
  act(obs, extra) {
    const b = this.b, HE = '_orig_actor.history_encoder.';
    // text features of the last 4 history frames
    for (let f = 0; f < 4; f++) {
      linear(...this.L('_orig_actor.text_feat_encoder.0'), obs, 113 + (6 + f) * 93, b.t1, 93, 128, elu);
      linear(...this.L('_orig_actor.text_feat_encoder.2'), b.t1, 0, b.t2, 128, 16, elu);
      b.tcat.set(b.t2, f * 16);
    }
    linear(...this.L('_orig_actor.text_feat_merger.0'), b.tcat, 0, b.text, 64, 16, elu);
    // history encoder: per-frame 93->30, conv 30->20 (k4 s2), conv 20->10 (k2 s1), 30->20
    for (let t = 0; t < 10; t++) {
      linear(...this.L(HE + 'encoder.0'), obs, 113 + t * 93, b.row, 93, 30, elu);
      for (let c = 0; c < 30; c++) b.proj[c * 10 + t] = b.row[c];
    }
    const conv = (W, bias, x, out, cin, len, cout, k, s) => {
      const lo = Math.floor((len - k) / s) + 1;
      for (let c = 0; c < cout; c++) for (let t = 0; t < lo; t++) {
        let acc = bias[c];
        for (let ci = 0; ci < cin; ci++) for (let j = 0; j < k; j++) acc += W[(c * cin + ci) * k + j] * x[ci * len + t * s + j];
        out[c * lo + t] = elu(acc);
      }
      return lo;
    };
    const l1 = conv(...this.L(HE + 'conv_layers.0'), b.proj, b.c1, 30, 10, 20, 4, 2);
    conv(...this.L(HE + 'conv_layers.2'), b.c1, b.c2, 20, l1, 10, 2, 1);
    linear(...this.L(HE + 'linear_output.0'), b.c2, 0, b.hl, 30, 20, elu);
    // student backbone on [extra | text | prop | demo | priv | hist latent]
    const x = b.x;
    x.set(extra, 0); x.set(b.text, 2325);
    for (let i = 0; i < 113; i++) x[2341 + i] = obs[i];
    x.set(b.hl, 2454);
    linear(...this.L('student_actor_backbone.0'), x, 0, b.h1, 2474, 1024, elu);
    linear(...this.L('student_actor_backbone.2'), b.h1, 0, b.h2, 1024, 1024, elu);
    linear(...this.L('student_actor_backbone.4'), b.h2, 0, b.h3, 1024, 512, elu);
    return linear(...this.L('student_actor_backbone.6'), b.h3, 0, b.a, 512, 15);
  }
}
