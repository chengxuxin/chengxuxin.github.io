// Reference-motion sampling (port of GMT MotionLib.calc_motion_frame + HumanoidEnv._get_mimic_obs).
// Per-frame layout (36 floats): root_pos3 | root_rot4 (xyzw) | root_vel3 | root_ang_vel3 | dof_pos23

export const TAR_STEPS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95];
export const CONTROL_DT = 0.02;
const F = 36;

function slerp(q0, q1, t, out) {
  let c = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
  const s1 = c < 0 ? -1 : 1;
  c = Math.abs(c);
  if (c >= 1) { for (let i = 0; i < 4; i++) out[i] = q0[i]; return out; }
  const half = Math.acos(c), sinHalf = Math.sqrt(1 - c * c);
  if (Math.abs(sinHalf) < 0.001) { for (let i = 0; i < 4; i++) out[i] = 0.5 * q0[i] + 0.5 * s1 * q1[i]; return out; }
  const a = Math.sin((1 - t) * half) / sinHalf, b = Math.sin(t * half) / sinHalf;
  for (let i = 0; i < 4; i++) out[i] = a * q0[i] + b * s1 * q1[i];
  return out;
}

// q in xyzw
export function quatRotateInverse(q, v, out) {
  const [x, y, z, w] = q;
  const a = 2 * w * w - 1, d = 2 * (x * v[0] + y * v[1] + z * v[2]);
  const cx = y * v[2] - z * v[1], cy = z * v[0] - x * v[2], cz = x * v[1] - y * v[0];
  out[0] = v[0] * a - cx * w * 2 + x * d;
  out[1] = v[1] * a - cy * w * 2 + y * d;
  out[2] = v[2] * a - cz * w * 2 + z * d;
  return out;
}

export class MotionRef {
  constructor(buf, fps) {
    this.buf = buf; this.fps = fps;
    this.n = buf.length / F;
    this.length = (this.n - 1) / fps;
    this.rot = new Float64Array(4); this.v = new Float64Array(3); this.w = new Float64Array(3);
    this.tmp = new Float64Array(3);
  }

  // First `seconds` of a clip (whole clip if it is shorter)
  trimmed(seconds) {
    const n = Math.min(this.n, Math.round(seconds * this.fps) + 1);
    return new MotionRef(this.buf.subarray(0, n * F), this.fps);
  }

  // A still reference holding frame `index` (negative counts from the end), with zero velocities
  static hold(src, index, seconds = 2) {
    const i = index < 0 ? src.n + index : index, n = Math.max(2, Math.round(seconds * src.fps) + 1);
    const buf = new Float32Array(n * F), fr = src.buf.subarray(i * F, (i + 1) * F);
    for (let k = 0; k < n; k++) {
      buf.set(fr, k * F);
      buf.fill(0, k * F + 7, k * F + 13);   // root vel + ang vel
    }
    return new MotionRef(buf, src.fps);
  }

  // Samples the frame at `time` into `out` (root_pos3, root_rot4 xyzw, vel3, angvel3, dof23 = 36)
  sample(time, out) {
    const { buf, n, length } = this;
    time -= Math.floor(time / length) * length;
    const phase = Math.min(1, Math.max(0, time / length));
    const i0 = Math.floor(phase * (n - 1)), i1 = Math.min(i0 + 1, n - 1);
    const t = phase * (n - 1) - i0, a = i0 * F, b = i1 * F;
    for (let k = 0; k < 3; k++) out[k] = (1 - t) * buf[a + k] + t * buf[b + k];
    slerp(buf.subarray(a + 3, a + 7), buf.subarray(b + 3, b + 7), t, this.rot);
    for (let k = 0; k < 4; k++) out[3 + k] = this.rot[k];
    for (let k = 7; k < 13; k++) out[k] = buf[a + k];
    for (let k = 13; k < 36; k++) out[k] = (1 - t) * buf[a + k] + t * buf[b + k];
    return out;
  }

  // 20 future frames x 30: z | roll pitch | local root vel3 | local yaw rate | dof23
  mimicObs(step, out) {
    const fr = new Float64Array(F);
    for (let s = 0; s < TAR_STEPS.length; s++) {
      this.sample((step + TAR_STEPS[s]) * CONTROL_DT, fr);
      const [x, y, z, w] = [fr[3], fr[4], fr[5], fr[6]], q = [x, y, z, w];
      const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
      const pitch = Math.asin(Math.min(1, Math.max(-1, 2 * (w * y - z * x))));
      quatRotateInverse(q, fr.subarray(7, 10), this.v);
      quatRotateInverse(q, fr.subarray(10, 13), this.w);
      const o = s * 30;
      out[o] = fr[2]; out[o + 1] = roll; out[o + 2] = pitch;
      out[o + 3] = this.v[0]; out[o + 4] = this.v[1]; out[o + 5] = this.v[2];
      out[o + 6] = this.w[2];
      for (let k = 0; k < 23; k++) out[o + 7 + k] = fr[13 + k];
    }
    return out;
  }
}
