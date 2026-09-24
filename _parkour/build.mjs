// Bundles the widget into ../assets/parkour/. Run: npm install && npm run build
import { build } from 'esbuild';
import { copyFileSync, mkdirSync } from 'node:fs';

const out = new URL('../assets/parkour/', import.meta.url).pathname;
const common = { bundle: true, format: 'esm', minify: true, target: 'es2020', legalComments: 'none' };

await build({ ...common, entryPoints: ['src/demo.js'], outfile: out + 'parkour-demo.js' });
// the worker imports the MuJoCo runtime shared with the other live demos (assets/mujoco/)
await build({ ...common, entryPoints: ['src/sim-worker.js'], outfile: out + 'parkour-sim.js', external: ['../mujoco/mujoco.js'] });
const mj = new URL('../assets/mujoco/', import.meta.url).pathname;
mkdirSync(mj, { recursive: true });
for (const f of ['mujoco.js', 'mujoco.wasm']) copyFileSync(new URL(`node_modules/@mujoco/mujoco/${f}`, import.meta.url), mj + f);
console.log('built ->', out);
