// GMT policy forward pass in plain JS (port of the TorchScript HardwareRefNN).
// obs (2154) = mimic 20x30 | proprio 74 | proprio history 20x74  ->  action (23)

const silu = (x) => x / (1 + Math.exp(-x));

// IEEE half -> float (weights ship as fp16 to halve the download)
export function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

function linear(W, b, x, out, nIn, nOut) {
  for (let o = 0; o < nOut; o++) {
    let s = b[o];
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
}

// x: channel-major [cin][L]  ->  out: channel-major [cout][Lout]
function conv1d(W, b, x, out, cin, L, cout, k, stride) {
  const Lout = Math.floor((L - k) / stride) + 1;
  for (let c = 0; c < cout; c++) {
    for (let t = 0; t < Lout; t++) {
      let s = b[c];
      for (let ci = 0; ci < cin; ci++) {
        const w = (c * cin + ci) * k, xi = ci * L + t * stride;
        for (let j = 0; j < k; j++) s += W[w + j] * x[xi + j];
      }
      out[c * Lout + t] = silu(s);
    }
  }
  return Lout;
}

export class GMTPolicy {
  constructor(manifest, weights) {
    this.p = {};
    for (const e of manifest.params) {
      const n = e.shape.reduce((a, b) => a * b, 1);
      this.p[e.name] = weights.subarray(e.offset, e.offset + n);
    }
    this.eps = manifest.eps;
    this.norm = new Float32Array(2154);
    this.backboneIn = new Float32Array(296);
    this.h1 = new Float32Array(1024); this.h2 = new Float32Array(1024);
    this.h3 = new Float32Array(512); this.h4 = new Float32Array(256);
    this.action = new Float32Array(23);
  }

  // Weights ship as fp16; the observation normalizer as fp32 (it is precision-sensitive).
  static fromBuffers(manifest, w16, norm) {
    const w = f16ToF32(new Uint16Array(w16)), n = new Float32Array(norm), half = n.length / 2;
    for (const e of manifest.params) {
      if (e.name === 'normalizer._mean') w.set(n.subarray(0, half), e.offset);
      if (e.name === 'normalizer._std') w.set(n.subarray(half), e.offset);
    }
    return new GMTPolicy(manifest, w);
  }

  static async load(base) {
    const get = (f) => fetch(base + f);
    const [manifest, w16, norm] = await Promise.all([
      get('policy.json').then((r) => r.json()), get('policy16.bin').then((r) => r.arrayBuffer()), get('policy_norm.bin').then((r) => r.arrayBuffer()),
    ]);
    return GMTPolicy.fromBuffers(manifest, w16, norm);
  }

  // 20 rows of `dIn` -> Linear(dIn->dH)+SiLU -> Conv(k6,s2)+SiLU -> Conv(k4,s2)+SiLU -> Linear(->dOut)
  encode(prefix, x, dIn, dH, c1, c2, dOut, out) {
    const p = this.p, rows = 20;
    const h = new Float32Array(rows * dH), cm = new Float32Array(dH * rows), tmp = new Float32Array(dH);
    for (let r = 0; r < rows; r++) {
      linear(p[prefix + '.encoder.0.weight'], p[prefix + '.encoder.0.bias'], x.subarray(r * dIn, (r + 1) * dIn), tmp, dIn, dH);
      for (let j = 0; j < dH; j++) h[r * dH + j] = silu(tmp[j]);
    }
    for (let r = 0; r < rows; r++) for (let j = 0; j < dH; j++) cm[j * rows + r] = h[r * dH + j]; // permute -> [ch][t]
    const a = new Float32Array(c1 * 8), b = new Float32Array(c2 * 3);
    const L1 = conv1d(p[prefix + '.conv_layers.0.weight'], p[prefix + '.conv_layers.0.bias'], cm, a, dH, rows, c1, 6, 2);
    conv1d(p[prefix + '.conv_layers.2.weight'], p[prefix + '.conv_layers.2.bias'], a, b, c1, L1, c2, 4, 2);
    linear(p[prefix + '.linear_output.weight'], p[prefix + '.linear_output.bias'], b, out, c2 * 3, dOut);
  }

  act(obs) {
    const p = this.p, n = this.norm, mean = p['normalizer._mean'], std = p['normalizer._std'];
    for (let i = 0; i < 2154; i++) n[i] = (obs[i] - mean[i]) / (std[i] + this.eps);

    const x = this.backboneIn;
    x.set(n.subarray(600, 674), 0);          // proprio
    x.set(n.subarray(0, 30), 74);            // current mimic frame
    this.encode('motion_encoder', n.subarray(0, 600), 30, 60, 40, 20, 128, x.subarray(104, 232));
    this.encode('history_encoder', n.subarray(674, 2154), 74, 30, 20, 10, 64, x.subarray(232, 296));

    const bb = (i) => [p[`actor_backbone.${i}.weight`], p[`actor_backbone.${i}.bias`]];
    linear(...bb(0), x, this.h1, 296, 1024); for (let i = 0; i < 1024; i++) this.h1[i] = silu(this.h1[i]);
    linear(...bb(2), this.h1, this.h2, 1024, 1024); for (let i = 0; i < 1024; i++) this.h2[i] = silu(this.h2[i]);
    linear(...bb(4), this.h2, this.h3, 1024, 512); for (let i = 0; i < 512; i++) this.h3[i] = silu(this.h3[i]);
    linear(...bb(6), this.h3, this.h4, 512, 256);
    // LayerNorm(256, eps=1e-5) + SiLU
    const h4 = this.h4, [g, beta] = bb(7);
    let mu = 0; for (let i = 0; i < 256; i++) mu += h4[i]; mu /= 256;
    let v = 0; for (let i = 0; i < 256; i++) v += (h4[i] - mu) ** 2; v /= 256;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < 256; i++) h4[i] = silu((h4[i] - mu) * inv * g[i] + beta[i]);
    linear(...bb(9), h4, this.action, 256, 23);
    return this.action;
  }
}
