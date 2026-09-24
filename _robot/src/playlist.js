// Plays GMT reference clips in a shuffled order with a short natural stand between them.
import { MotionRef, CONTROL_DT } from './motion.js';

const IDLE_MIN = 1.5, IDLE_MAX = 2.5;

export class Playlist {
  // manifest: data/motions.json; bufs: { name: Float32Array }
  constructor(manifest, bufs, random = Math.random) {
    this.random = random;
    this.clips = manifest.clips.map((c) => {
      const ref = new MotionRef(bufs[c.name], c.fps);
      return { name: c.name, ref: c.seconds ? ref.trimmed(c.seconds) : ref };
    });
    const src = this.clips.find((c) => c.name === manifest.idle.from).ref;
    this.idle = MotionRef.hold(src, manifest.idle.frame, IDLE_MAX + 1);
    this.queue = [];
    this.last = null;
  }

  nextClip() {
    if (!this.queue.length) {                        // reshuffle; never repeat across the seam
      this.queue = this.clips.slice().sort(() => this.random() - 0.5);
      if (this.queue[0] === this.last && this.queue.length > 1) this.queue.push(this.queue.shift());
    }
    return (this.last = this.queue.shift());
  }

  // Starts with a stand, then alternates clip / stand.
  start(ctl) {
    this.enter(ctl, { name: 'stand', ref: this.idle, until: IDLE_MIN });
  }

  enter(ctl, item) {
    this.current = item;
    ctl.setMotion(item.ref);
  }

  // Call once per control step (or more often): switches when the current item has played out.
  update(ctl) {
    const t = ctl.step * CONTROL_DT, cur = this.current;
    const end = cur.name === 'stand' ? cur.until : cur.ref.length;
    if (t < end) return;
    if (cur.name === 'stand') this.enter(ctl, { ...this.nextClip() });
    else this.enter(ctl, { name: 'stand', ref: this.idle, until: IDLE_MIN + this.random() * (IDLE_MAX - IDLE_MIN) });
  }
}
