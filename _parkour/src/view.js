// Renderer for the Extreme Parkour live demo (three.js).
//   1  id pass -> offscreen: link id (robot) or surface normal (terrain) + depth
//   2  visible pass: terrain faces flat-shaded in warm greys, robot filled with paper colour
//   3  terrain dots
//   4  one full-screen ink pass for every line, so all lines share one width:
//        robot   - nearer side of a link-id boundary
//        terrain - nearer side of a depth jump, and the upper side of a crease (normal change) between
//                  faces that touch; each edge is drawn on one side only, so no line doubles up
import * as THREE from 'three';
import { HS } from './terrain.js';

export const COLORS = { ink: 0x23231f, bg: 0xfbfbf9, accent: 0xc2410c, hair: 0xb9b8b0 };
const LAYERS = { id: 1, vis: 2, dots: 3 };
const SLOPE_T = 1.5 * HS;
const toThree = (x, y, z, v = new THREE.Vector3()) => v.set(x, z, -y);     // MuJoCo z-up -> three y-up

// ---------------------------------------------------------------- robot
export class LineArtRobot {
  constructor(vis, bin) {
    const idMats = vis.bodies.map((_, k) => new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB((k + 1) / 16, 0.5, 0.5, THREE.LinearSRGBColorSpace), side: THREE.DoubleSide }));
    const fillMat = new THREE.MeshBasicMaterial({ color: COLORS.bg, side: THREE.DoubleSide });
    this.group = new THREE.Group();
    this.bodies = vis.bodies.map(() => { const g = new THREE.Group(); this.group.add(g); return g; });
    // go2_vis.bin: int16 positions (x vis.scale, in body frames), then uint16 per-geom indices
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
  // poses: per body pos3 + quat4 (wxyz), in MuJoCo world coordinates (the stage's world group converts)
  setPoses(p) {
    this.bodies.forEach((b, k) => { const o = k * 7; b.position.set(p[o], p[o + 1], p[o + 2]); b.quaternion.set(p[o + 4], p[o + 5], p[o + 6], p[o + 3]); });
  }
}

// ---------------------------------------------------------------- terrain
const WORLD_VERT = /* glsl */ `
  varying vec3 vWp;
  void main() { vec4 wp = modelMatrix * vec4(position, 1.0); vWp = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`;

// id pass: r = 0 (terrain), gba = flat face normal (three world, y up)
const ID_FRAG = /* glsl */ `
  varying vec3 vWp;
  void main() { vec3 n = normalize(cross(dFdx(vWp), dFdy(vWp))); gl_FragColor = vec4(0.0, n * 0.5 + 0.5); }`;

// Flat shading like an architectural axonometric: pale tops, walls in two greys by orientation,
// everything darker the deeper it sits (pits read at a glance), fading to paper with distance.
const FACE_FRAG = /* glsl */ `
  uniform vec3 bg, top, wallLit, wallShade, deep; uniform float fog0, fog1;
  varying vec3 vWp;
  void main() {
    vec3 n = normalize(cross(dFdx(vWp), dFdy(vWp)));
    float lam = dot(n, normalize(vec3(-0.7, 0.6, 0.25)));
    vec3 c = n.y > 0.6 ? top : mix(wallShade, wallLit, smoothstep(0.2, 0.5, lam));
    c = mix(c, deep, clamp(-vWp.y, 0.0, 1.0) * 0.65);
    c = mix(c, bg, smoothstep(fog0, fog1, distance(cameraPosition, vWp)));
    gl_FragColor = vec4(c, 1.0);
    #include <colorspace_fragment>
  }`;

// Render surface: the noise-free grid, meshed cell by cell exactly like the physics reads it
// (Track.height): a cell whose corners span more than SLOPE_T is flat at its lowest corner,
// otherwise bilinear; vertical walls fill every height step between neighbouring cells.
// A big flat apron surrounds the strip.
export class TerrainView {
  constructor() {
    this.group = new THREE.Group();
    this.idMat = new THREE.ShaderMaterial({ vertexShader: WORLD_VERT, fragmentShader: ID_FRAG, side: THREE.DoubleSide });
    const C = (hex) => ({ value: new THREE.Color(hex) });
    this.faceMat = new THREE.ShaderMaterial({
      vertexShader: WORLD_VERT, fragmentShader: FACE_FRAG, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      uniforms: {
        bg: C(COLORS.bg), top: C(0xf3f2ed), wallLit: C(0xe5e3db), wallShade: C(0xd5d3c9), deep: C(0xc4c1b6),
        fog0: { value: 10 }, fog1: { value: 24 },
      },
    });
    this.dotMat = new THREE.PointsMaterial({ color: COLORS.hair, size: 1.6, sizeAttenuation: false, transparent: true, depthWrite: false });
    this.center = new THREE.Vector2();
    this.dotMat.onBeforeCompile = (sh) => {                  // dots fade out around the robot
      sh.uniforms.center = { value: this.center };
      sh.vertexShader = 'uniform vec2 center; varying float vFade;\n' + sh.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvec4 wp = modelMatrix * vec4(position, 1.0);\nvFade = 1.0 - smoothstep(1.5, 5.0, length(wp.xz - center));');
      sh.fragmentShader = 'varying float vFade;\n' + sh.fragmentShader.replace('#include <premultiplied_alpha_fragment>',
        'gl_FragColor.a *= vFade;\n#include <premultiplied_alpha_fragment>');
    };
  }

  build(track) {
    for (const o of [...this.group.children]) { this.group.remove(o); o.geometry.dispose(); }
    const nx = Math.min(track.cols, (track.used ?? track.cols - 1) + 40), ny = 51;
    // noise-free grid: clean walls and flat floors (the roughness is only a few cm)
    const R = track.C.subarray(0, nx * ny);
    const X = (a) => track.x0 + a * HS, Y = (b) => -2 + b * HS;
    // per-cell corner heights (00, 10, 11, 01) after the slope rule
    const Q = new Float64Array((nx - 1) * (ny - 1) * 4);
    for (let a = 0; a < nx - 1; a++) for (let b = 0; b < ny - 1; b++) {
      const h = [R[a * ny + b], R[(a + 1) * ny + b], R[(a + 1) * ny + b + 1], R[a * ny + b + 1]];
      const lo = Math.min(...h), flat = Math.max(...h) - lo > SLOPE_T, o = (a * (ny - 1) + b) * 4;
      for (let k = 0; k < 4; k++) Q[o + k] = flat ? lo : h[k];
    }
    const pos = [], idx = [];
    const quad = (p) => { const i = pos.length / 3; pos.push(...p); idx.push(i, i + 1, i + 2, i, i + 2, i + 3); };
    for (let a = 0; a < nx - 1; a++) for (let b = 0; b < ny - 1; b++) {
      const o = (a * (ny - 1) + b) * 4, x0 = X(a), x1 = X(a + 1), y0 = Y(b), y1 = Y(b + 1);
      quad([x0, y0, Q[o], x1, y0, Q[o + 1], x1, y1, Q[o + 2], x0, y1, Q[o + 3]]);
      if (a < nx - 2) {                                  // wall on the +x edge
        const p = ((a + 1) * (ny - 1) + b) * 4;
        if (Math.abs(Q[o + 1] - Q[p]) > 1e-4 || Math.abs(Q[o + 2] - Q[p + 3]) > 1e-4) quad([x1, y0, Q[o + 1], x1, y1, Q[o + 2], x1, y1, Q[p + 3], x1, y0, Q[p]]);
      }
      if (b < ny - 2) {                                  // wall on the +y edge
        const p = o + 4;
        if (Math.abs(Q[o + 3] - Q[p]) > 1e-4 || Math.abs(Q[o + 2] - Q[p + 1]) > 1e-4) quad([x0, y1, Q[o + 3], x1, y1, Q[o + 2], x1, y1, Q[p + 1], x0, y1, Q[p]]);
      }
    }
    // apron: flat ground around the strip, far enough that its edge is never in view
    const E = 60, xa = X(0), xb = X(nx - 1);
    for (const [u0, u1, v0, v1] of [[xa - E, xb + E, -2 - E, -2], [xa - E, xb + E, 2, 2 + E], [xa - E, xa, -2, 2], [xb, xb + E, -2, 2]])
      quad([u0, v0, 0, u1, v0, 0, u1, v1, 0, u0, v1, 0]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    const id = new THREE.Mesh(geo, this.idMat); id.layers.set(LAYERS.id);
    const faces = new THREE.Mesh(geo, this.faceMat); faces.layers.set(LAYERS.vis);
    // surface dots every 2 vertices (strip) and on the apron nearby
    const dots = [];
    for (let a = -30; a < nx + 30; a += 2) for (let b = -39; b < ny + 40; b += 2) {
      const inside = a >= 0 && a < nx && b >= 0 && b < ny;
      dots.push(X(a), Y(b), (inside ? R[a * ny + b] : 0) + 0.004);
    }
    const dgeo = new THREE.BufferGeometry(); dgeo.setAttribute('position', new THREE.Float32BufferAttribute(dots, 3));
    const pts = new THREE.Points(dgeo, this.dotMat); pts.layers.set(LAYERS.dots);
    this.group.add(id, faces, pts);
  }

  follow(x, y) { this.center.set(x, -y); }
}

// ---------------------------------------------------------------- stage
const INK_FRAG = /* glsl */ `
  uniform sampler2D tDepth, tId; uniform vec2 texel; uniform float near, far, radius, thresh, fade0, fade1, alpha;
  uniform vec3 ink; varying vec2 vUv;
  float lin(float d) { float z = d * 2.0 - 1.0; return 2.0 * near * far / (far + near - z * (far - near)); }
  void main() {
    vec4 C = texture2D(tId, vUv);
    if (C.r > 0.97) discard;                                  // background
    float dc = lin(texture2D(tDepth, vUv).x), hits = 0.0;
    bool robot = C.r > 0.03;
    vec3 nc = C.gba * 2.0 - 1.0;
    for (int i = 0; i < 8; i++) {
      vec2 dir = vec2(cos(float(i) * 0.785398), sin(float(i) * 0.785398)) * texel * radius;
      for (int j = 1; j <= 2; j++) {
        vec2 o = dir * (j == 1 ? 0.5 : 1.0);
        vec4 N = texture2D(tId, vUv + o);
        float dn = lin(texture2D(tDepth, vUv + o).x);
        if (robot) {
          if (abs(N.r - C.r) > 0.03 && dn > dc * 1.004) hits += 1.0;       // links grazing each other draw nothing
        } else {
          float dp = lin(texture2D(tDepth, vUv - o).x);
          if (dn > dc && (dn + dp - 2.0 * dc) / dc > thresh) hits += 1.0;  // depth jump beyond the local plane
          else if (N.r < 0.03 && abs(dn - dc) < thresh * dc) {             // crease between touching faces (not across
            vec3 nn = N.gba * 2.0 - 1.0;                                    // a depth jump, which the rule above draws), upper side only
            if (dot(nc, nn) < 0.8 && (nc.y > nn.y + 0.02 || (abs(nc.y - nn.y) <= 0.02 && nc.x + nc.z > nn.x + nn.z))) hits += 1.0;
          }
        }
      }
    }
    float k = min(hits / 3.0, 1.0) * (1.0 - smoothstep(fade0, fade1, dc));
    gl_FragColor = vec4(ink, k * alpha);
    #include <colorspace_fragment>
  }`;

export class Stage {
  constructor(canvas, { fov = 26, distance = 4.6, azimuth = -1.9, elevation = 0.34, lead = 0.55, line = 1.2, thresh = 0.02 } = {}) {
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.autoClear = false;
    Object.assign(this, { distance, azimuth, elevation, lead, line });
    this.scene = new THREE.Scene();
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);
    this.fov0 = fov;
    this.camera = new THREE.PerspectiveCamera(fov, 1, 0.5, 60);
    this.target = new THREE.Vector3();               // MuJoCo coords
    this.rt = new THREE.WebGLRenderTarget(1, 1, { depthTexture: new THREE.DepthTexture(1, 1), minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.ink = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: this.rt.depthTexture }, tId: { value: this.rt.texture }, texel: { value: new THREE.Vector2() },
        near: { value: 0.5 }, far: { value: 60 }, radius: { value: line }, thresh: { value: thresh }, fade0: { value: 10 }, fade1: { value: 20 },
        ink: { value: new THREE.Color(COLORS.ink) }, alpha: { value: 0.92 },
      },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: INK_FRAG, depthTest: false, depthWrite: false, transparent: true,
    });
    this.quadScene = new THREE.Scene();
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.ink));
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
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

  follow(x, y, z, k = 0.06, snap = false) {
    const t = this.target, gx = x + this.lead, gy = y * 0.5, gz = z - 0.15;
    if (snap) { t.set(gx, gy, gz); return; }
    t.x += (gx - t.x) * k; t.y += (gy - t.y) * k; t.z += (gz - t.z) * k * 0.6;
  }

  render() {
    const { target: t, azimuth: az, elevation: el, distance: d, renderer: r, camera: cam } = this;
    cam.position.copy(toThree(t.x + d * Math.cos(el) * Math.cos(az), t.y + d * Math.cos(el) * Math.sin(az), t.z + d * Math.sin(el)));
    cam.lookAt(toThree(t.x, t.y, t.z));
    cam.layers.set(LAYERS.id);
    r.setRenderTarget(this.rt); r.setClearColor(0xffffff, 1); r.clear(); r.render(this.scene, cam);
    r.setRenderTarget(null); r.setClearColor(COLORS.bg, 1); r.clear();
    cam.layers.set(LAYERS.vis); r.render(this.scene, cam);
    cam.layers.set(LAYERS.dots); r.render(this.scene, cam);
    r.render(this.quadScene, this.quadCam);
  }
}
