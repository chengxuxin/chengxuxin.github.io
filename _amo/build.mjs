// Bundles the demo into ../assets/amo/. Run: npm install && npm run build
// data/ holds assets exported from AMO's release (github.com/OpenTeleVision/AMO, Apache-2.0): the policy
// (big layers int8), the physics-only G1 (.mjb) and decimated visual meshes
import { build } from 'esbuild';
import { copyFileSync, mkdirSync } from 'node:fs';

const out = new URL('../assets/amo/', import.meta.url).pathname;
const common = { bundle: true, format: 'esm', minify: true, target: 'es2020', legalComments: 'none' };

await build({ ...common, entryPoints: ['src/demo.js'], outfile: out + 'amo-demo.js' });
// the worker imports the MuJoCo runtime shared with the footer widgets (assets/mujoco/)
await build({ ...common, entryPoints: ['src/sim-worker.js'], outfile: out + 'amo-sim.js', external: ['../mujoco/mujoco.js'] });
const mj = new URL('../assets/mujoco/', import.meta.url).pathname;
mkdirSync(mj, { recursive: true });
for (const f of ['mujoco.js', 'mujoco.wasm']) copyFileSync(new URL(`node_modules/@mujoco/mujoco/${f}`, import.meta.url), mj + f);
console.log('built ->', out);
