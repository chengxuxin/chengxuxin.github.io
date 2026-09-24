// Random courses shared by the page and the headless tests: all 4 of the paper's terrain types
// in random order, each at a random difficulty.
import { Track, subTerrain, makeRng } from './terrain.js';

// Isaaclab_Parkour names -> label + the difficulty range we sample
export const TYPES = {
  parkour_gap: { label: 'gaps', d: [0.3, 0.85] },
  parkour_hurdle: { label: 'hurdles', d: [0.3, 0.85] },
  parkour_step: { label: 'steps', d: [0.3, 0.75] },
  parkour: { label: 'tilted ramps', d: [0.3, 0.85] },
};

export function makeCourse(seed) {
  const rng = makeRng(seed);
  const order = Object.keys(TYPES).map((t) => [rng.next(), t]).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
  return new Track(order.map((t) => subTerrain(t, rng.uniform(...TYPES[t].d), rng)));
}

// seconds without reaching a new waypoint before we call it stuck (waypoints are up to ~2.3 m apart)
export const stuckAfter = (vx) => 3 + 3.5 / vx;
