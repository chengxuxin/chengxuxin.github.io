// Renderer for the AMO demo (three.js), same passes as the parkour widget:
//   1  id pass -> offscreen: link id (robot) or surface normal (floor) + depth
//   2  visible pass: floor in warm grey fading to paper, robot filled with paper colour
//   3  floor dots
//   4  one full-screen ink pass for every line (robot: nearer side of a link-id boundary)
import * as THREE from 'three';

export const COLORS = { ink: 0x23231f, bg: 0xfbfbf9, accent: 0xc2410c, hair: 0xb9b8b0 };
const LAYERS = { id: 1, vis: 2, dots: 3 };
const toThree = (x, y, z, v = new THREE.Vector3()) => v.set(x, z, -y);     // MuJoCo z-up -> three y-up
// link k is drawn with id 0.1 + k / 64 in the red channel; floor 0, background 1
export const idOf = (r) => (r > 0.05 && r < 0.97 ? Math.round((r - 0.1) * 64) : -1);

export class LineArtRobot {
  constructor(vis, bin) {
    const idMats = vis.bodies.map((_, k) => new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(0.1 + k / 64, 0.5, 0.5, THREE.LinearSRGBColorSpace), side: THREE.DoubleSide }));
    const fillMat = new THREE.MeshBasicMaterial({ color: COLORS.bg, side: THREE.DoubleSide });
    this.group = new THREE.Group();
    this.bodies = vis.bodies.map(() => { const g = new THREE.Group(); this.group.add(g); return g; });
    // g1_vis.bin: int16 positions (x vis.scale, in body frames), then uint16 per-geom indices
    const V = new Int16Array(bin, 0, vis.nvert * 3), F = new Uint16Array(bin, vis.nvert * 6, vis.nface * 3);
    for (const g of vis.geoms) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(V.subarray(g.v0 * 3, (g.v0 + g.nv) * 3), (v) => v * vis.scale), 3));
      geo.setIndex(new THREE.BufferAttribute(F.slice(g.f0 * 3, (g.f0 + g.nf) * 3), 1));
      const id = new THREE.Mesh(geo, idMats[g.b]); id.layers.set(LAYERS.id);
      const fill = new THREE.Mesh(geo, fillMat); fill.layers.set(LAYERS.vis);
      this.bodies[g.b].add(id, fill);
    }
  }
  // poses: per body pos3 + quat4 (wxyz), MuJoCo world coordinates
  setPoses(p) {
    this.bodies.forEach((b, k) => { const o = k * 7; b.position.set(p[o], p[o + 1], p[o + 2]); b.quaternion.set(p[o + 4], p[o + 5], p[o + 6], p[o + 3]); });
  }
}

// Ground: a large plane in the same warm grey as the parkour terrain tops, with a dot grid that
// fades out around the robot.
const WORLD_VERT = /* glsl */ `
  varying vec3 vWp;
  void main() { vec4 wp = modelMatrix * vec4(position, 1.0); vWp = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`;
const ID_FRAG = /* glsl */ `void main() { gl_FragColor = vec4(0.0, 0.5, 1.0, 0.5); }`;       // id 0, normal +y
const FLOOR_FRAG = /* glsl */ `
  uniform vec3 bg, top; uniform float fog0, fog1; varying vec3 vWp;
  void main() {
    gl_FragColor = vec4(mix(top, bg, smoothstep(fog0, fog1, distance(cameraPosition, vWp))), 1.0);
    #include <colorspace_fragment>
  }`;

export class Floor {
  constructor() {
    this.group = new THREE.Group();
    const plane = new THREE.PlaneGeometry(200, 200);
    const id = new THREE.Mesh(plane, new THREE.ShaderMaterial({ vertexShader: WORLD_VERT, fragmentShader: ID_FRAG, side: THREE.DoubleSide }));
    const C = (hex) => ({ value: new THREE.Color(hex) });
    const vis = new THREE.Mesh(plane, new THREE.ShaderMaterial({
      vertexShader: WORLD_VERT, fragmentShader: FLOOR_FRAG, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      uniforms: { bg: C(COLORS.bg), top: C(0xf3f2ed), fog0: { value: 5 }, fog1: { value: 14 } },
    }));
    id.layers.set(LAYERS.id); vis.layers.set(LAYERS.vis);
    this.center = new THREE.Vector2();
    const S = 0.2, N = 40, dots = [];
    for (let i = -N; i <= N; i++) for (let j = -N; j <= N; j++) dots.push(i * S, j * S, 0.002);
    const dgeo = new THREE.BufferGeometry(); dgeo.setAttribute('position', new THREE.Float32BufferAttribute(dots, 3));
    const mat = new THREE.PointsMaterial({ color: COLORS.hair, size: 1.8, sizeAttenuation: false, transparent: true, depthWrite: false });
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.center = { value: this.center };
      sh.vertexShader = 'uniform vec2 center; varying float vFade;\n' + sh.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvec4 wp = modelMatrix * vec4(position, 1.0);\nvFade = 1.0 - smoothstep(1.0, 3.2, length(wp.xz - center));');
      sh.fragmentShader = 'varying float vFade;\n' + sh.fragmentShader.replace('#include <premultiplied_alpha_fragment>',
        'gl_FragColor.a *= vFade;\n#include <premultiplied_alpha_fragment>');
    };
    this.dots = new THREE.Points(dgeo, mat); this.dots.layers.set(LAYERS.dots);
    this.S = S;
    this.group.add(id, vis, this.dots);
  }
  follow(x, y) {                                    // keep the finite dot grid under the robot, snapped to the grid
    this.dots.position.set(Math.round(x / this.S) * this.S, Math.round(y / this.S) * this.S, 0);
    this.center.set(x, -y);
  }
}

const INK_FRAG = /* glsl */ `
  uniform sampler2D tDepth, tId; uniform vec2 texel; uniform float near, far, radius, fade0, fade1, alpha;
  uniform vec3 ink, robotInk; varying vec2 vUv;
  float lin(float d) { float z = d * 2.0 - 1.0; return 2.0 * near * far / (far + near - z * (far - near)); }
  void main() {
    vec4 C = texture2D(tId, vUv);
    if (C.r < 0.05 || C.r > 0.97) discard;                    // lines are drawn on the robot side only
    float dc = lin(texture2D(tDepth, vUv).x), hits = 0.0;
    for (int i = 0; i < 8; i++) {
      vec2 dir = vec2(cos(float(i) * 0.785398), sin(float(i) * 0.785398)) * texel * radius;
      for (int j = 1; j <= 2; j++) {
        vec2 o = dir * (j == 1 ? 0.5 : 1.0);
        float idn = texture2D(tId, vUv + o).r, dn = lin(texture2D(tDepth, vUv + o).x);
        if (abs(idn - C.r) > 0.008 && dn > dc * 1.004) hits += 1.0;       // links grazing each other draw nothing
      }
    }
    float k = min(hits / 3.0, 1.0) * (1.0 - smoothstep(fade0, fade1, dc));
    gl_FragColor = vec4(robotInk, k * alpha);
    #include <colorspace_fragment>
  }`;

export class Stage {
  constructor(canvas, { fov = 24, distance = 4.2, azimuth = 0.75, elevation = 0.16, height = 0.62, line = 1.25 } = {}) {
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.autoClear = false;
    Object.assign(this, { distance, azimuth, elevation, line });
    this.scene = new THREE.Scene();
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);
    this.fov0 = fov;
    this.camera = new THREE.PerspectiveCamera(fov, 1, 0.3, 40);
    this.target = new THREE.Vector3(0, 0, height);     // MuJoCo coords
    this.rt = new THREE.WebGLRenderTarget(1, 1, { depthTexture: new THREE.DepthTexture(1, 1), minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.robotInk = new THREE.Color(COLORS.ink);
    this.ink = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: this.rt.depthTexture }, tId: { value: this.rt.texture }, texel: { value: new THREE.Vector2() },
        near: { value: 0.3 }, far: { value: 40 }, radius: { value: line }, fade0: { value: 12 }, fade1: { value: 20 },
        ink: { value: new THREE.Color(COLORS.ink) }, robotInk: { value: this.robotInk }, alpha: { value: 0.92 },
      },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: INK_FRAG, depthTest: false, depthWrite: false, transparent: true,
    });
    this.quadScene = new THREE.Scene();
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.ink));
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.px = new Uint8Array(4);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const c = this.renderer.domElement, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    const pr = this.renderer.getPixelRatio();
    this.renderer.setSize(w, h, false);
    this.rt.setSize(Math.round(w * pr), Math.round(h * pr));
    this.ink.uniforms.texel.value.set(1 / (w * pr), 1 / (h * pr));
    this.ink.uniforms.radius.value = this.line * pr;
    this.camera.aspect = w / h;
    // portrait (phones): open the vertical field of view so the horizontal one doesn't get too narrow
    this.camera.fov = w >= h ? this.fov0 : (360 / Math.PI) * Math.atan(Math.tan((this.fov0 * Math.PI) / 360) / (w / h) ** 0.75);
    this.camera.updateProjectionMatrix();
  }

  // 0 = ink, 1 = accent (robot outline while it can be / is being grabbed)
  setRobotTint(k) { this.robotInk.set(COLORS.ink).lerp(new THREE.Color(COLORS.accent), k); }

  follow(x, y, k = 0.05) { this.target.x += (x - this.target.x) * k; this.target.y += (y - this.target.y) * k; }

  // link index under a canvas pixel (css px), from the last id pass; -1 = none
  pick(cx, cy) {
    const c = this.renderer.domElement, pr = this.renderer.getPixelRatio();
    const x = Math.floor(cx * pr), y = Math.floor((c.clientHeight - cy) * pr);
    if (x < 0 || y < 0 || x >= this.rt.width || y >= this.rt.height) return -1;
    this.renderer.readRenderTargetPixels(this.rt, x, y, 1, 1, this.px);
    return idOf(this.px[0] / 255);
  }

  // world-space (MuJoCo) directions of screen right / up, and metres per css pixel at the depth of
  // point p (MuJoCo coords; default: the orbit target)
  screenBasis(p) {
    const e = this.camera.matrixWorld.elements, h = this.renderer.domElement.clientHeight, c = this.camera.position;
    const right = [e[0], -e[2], e[1]], up = [e[4], -e[6], e[5]];         // three (x, y, z) -> MuJoCo (x, -z, y)
    const depth = p ? -((p[0] - c.x) * e[8] + (p[2] - c.y) * e[9] + (-p[1] - c.z) * e[10]) : this.distance;
    return { right, up, mpp: (2 * depth * Math.tan((this.camera.fov * Math.PI) / 360)) / h };
  }

  toScreen(x, y, z) {
    const v = toThree(x, y, z).project(this.camera), c = this.renderer.domElement;
    return [(v.x + 1) / 2 * c.clientWidth, (1 - v.y) / 2 * c.clientHeight];
  }

  render() {
    const { target: t, azimuth: az, elevation: el, distance: d, renderer: r, camera: cam } = this;
    cam.position.copy(toThree(t.x + d * Math.cos(el) * Math.cos(az), t.y + d * Math.cos(el) * Math.sin(az), t.z + d * Math.sin(el)));
    cam.lookAt(toThree(t.x, t.y, t.z));
    cam.updateMatrixWorld();
    cam.layers.set(LAYERS.id);
    r.setRenderTarget(this.rt); r.setClearColor(0xffffff, 1); r.clear(); r.render(this.scene, cam);
    r.setRenderTarget(null); r.setClearColor(COLORS.bg, 1); r.clear();
    cam.layers.set(LAYERS.vis); r.render(this.scene, cam);
    cam.layers.set(LAYERS.dots); r.render(this.scene, cam);
    r.render(this.quadScene, this.quadCam);
  }
}
