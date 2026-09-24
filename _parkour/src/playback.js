// Smooth playback of the poses the simulation worker posts. The worker sends a frame (sim time + body
// poses) every ~16-20 ms on its own timer, which beats against the display refresh (60 / 120 Hz): drawn
// as they arrive, the robot stalls on some display frames and jumps on others while the camera glides,
// which reads as a fast jitter. Instead, frames are buffered and played back ~40 ms (sim time) behind
// the newest one, interpolating positions linearly and orientations by normalised lerp.
const DELAY = 0.04;

export class PosePlayback {
  // bodies: number of bodies (7 floats each: pos3 + quat4 wxyz); extras: names of other numeric arrays to lerp
  constructor(bodies, extras = []) {
    Object.assign(this, { bodies, extras, buf: [], clock: null });
    this.out = { poses: new Float32Array(bodies * 7) };
  }

  reset() { this.buf.length = 0; this.clock = null; }

  push(t, poses, extra = {}) {
    const b = this.buf;
    if (b.length && t <= b[b.length - 1].t) {
      if (t === b[b.length - 1].t) return;
      this.reset();                                 // sim time went back: new course / reset, start over
    }
    b.push({ t, poses, ...extra });
    if (b.length > 8) b.shift();
  }

  // advance the playback clock by dt s of wall time at `rate` (sim s per wall s); null until frames arrive
  sample(dt, rate = 1) {
    const b = this.buf;
    if (!b.length) return null;
    const newest = b[b.length - 1].t, target = newest - DELAY * rate;
    if (this.clock === null || Math.abs(this.clock - target) > 0.25 * Math.max(rate, 0.3)) this.clock = target;
    else this.clock += dt * rate + (target - this.clock) * 0.08;     // follow wall time, drift toward the target
    const c = Math.min(Math.max(this.clock, b[0].t), newest);
    let i = 0;
    while (i < b.length - 2 && b[i + 1].t < c) i++;
    const A = b[i], B = b[Math.min(i + 1, b.length - 1)], a = B.t > A.t ? (c - A.t) / (B.t - A.t) : 1;
    const P = this.out.poses, pa = A.poses, pb = B.poses;
    for (let k = 0; k < this.bodies; k++) {
      const o = k * 7;
      for (let j = 0; j < 3; j++) P[o + j] = pa[o + j] + (pb[o + j] - pa[o + j]) * a;
      const s = pa[o + 3] * pb[o + 3] + pa[o + 4] * pb[o + 4] + pa[o + 5] * pb[o + 5] + pa[o + 6] * pb[o + 6] < 0 ? -1 : 1;
      let n = 0;
      for (let j = 3; j < 7; j++) { P[o + j] = pa[o + j] + (s * pb[o + j] - pa[o + j]) * a; n += P[o + j] * P[o + j]; }
      n = Math.sqrt(n);
      for (let j = 3; j < 7; j++) P[o + j] /= n;
    }
    for (const name of this.extras) this.out[name] = A[name].map((v, j) => v + (B[name][j] - v) * a);
    return this.out;
  }
}
