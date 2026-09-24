// Technical-illustration line-art renderer for the G1 (three.js).
//   pass 1  fills -> offscreen depth
//   pass 2  full-screen shader: ink contour wherever depth jumps (silhouettes + occlusion lines)
//   pass 3  fills -> depth + stencil only (so pass 4 is occluded, and the ghost stays outside the robot)
//   pass 4  thin crease lines, floor dots, ghost, overlays
import * as THREE from 'three';

export const COLORS = { ink: 0x1c1c1a, bg: 0xfbfbf9, accent: 0xc2410c, hair: 0xcfcec7 };
export const LAYERS = { fill: 1, line: 2, over: 3, ghost: 4 };

function bodyMeshes(vis, bin, make) {
  const group = new THREE.Group();
  const bodies = vis.bodies.map(() => { const g = new THREE.Group(); group.add(g); return g; });
  const V = new Float32Array(bin, 0, vis.nvert * 3), F = new Uint32Array(bin, vis.nvert * 12, vis.nface * 3);
  for (const g of vis.geoms) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(V.slice(g.v0 * 3, (g.v0 + g.nv) * 3), 3));
    geo.setIndex(new THREE.BufferAttribute(F.slice(g.f0 * 3, (g.f0 + g.nf) * 3), 1));
    bodies[g.b].add(...make(geo));
  }
  return { group, bodies };
}

// poses: per body pos3 + quat4 (wxyz, MuJoCo convention)
function setPoses(bodies, p) {
  for (let k = 0; k < bodies.length; k++) {
    const o = k * 7;
    bodies[k].position.set(p[o], p[o + 1], p[o + 2]);
    bodies[k].quaternion.set(p[o + 4], p[o + 5], p[o + 6], p[o + 3]);
  }
}

export class LineArtRobot {
  constructor(vis, bin, { crease = 52, creaseOpacity = 0.45 } = {}) {
    const fillMat = new THREE.MeshBasicMaterial({ color: COLORS.bg });
    const lineMat = (this.lineMat = new THREE.LineBasicMaterial({ color: COLORS.ink, transparent: true, opacity: creaseOpacity }));
    Object.assign(this, bodyMeshes(vis, bin, (geo) => {
      const fill = new THREE.Mesh(geo, fillMat); fill.layers.set(LAYERS.fill);
      const lines = new THREE.LineSegments(new THREE.EdgesGeometry(geo, crease), lineMat); lines.layers.set(LAYERS.line);
      return [fill, lines];
    }));
  }
  setPoses(p) { setPoses(this.bodies, p); }
}

// Translucent "intent" silhouette: drawn once per pixel, only outside the robot's own silhouette.
export class GhostRobot {
  constructor(vis, bin, { opacity = 0.12 } = {}) {
    const mat = new THREE.MeshBasicMaterial({
      color: COLORS.accent, transparent: true, opacity, depthTest: false, depthWrite: false,
      stencilWrite: true, stencilRef: 0, stencilFunc: THREE.EqualStencilFunc, stencilZPass: THREE.IncrementStencilOp,
    });
    Object.assign(this, bodyMeshes(vis, bin, (geo) => { const m = new THREE.Mesh(geo, mat); m.layers.set(LAYERS.ghost); return [m]; }));
  }
  setPoses(p) { setPoses(this.bodies, p); }
}

// World-fixed dot grid that fades out radially around the robot.
class DotFloor {
  constructor({ spacing = 0.18, half = 8, radius = 0.9, size = 1.8 } = {}) {
    this.spacing = spacing;
    const pos = [];
    for (let x = -half; x <= half; x++) for (let y = -half; y <= half; y++) pos.push(x * spacing, y * spacing, 0);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: COLORS.hair, size, sizeAttenuation: false, depthWrite: false, transparent: true });
    this.center = new THREE.Vector2();
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.center = { value: this.center }; sh.uniforms.radius = { value: radius };
      sh.vertexShader = 'uniform vec2 center; uniform float radius; varying float vFade;\n' + sh.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvec4 wp = modelMatrix * vec4(position, 1.0);\nvFade = 1.0 - smoothstep(radius * 0.3, radius, length(wp.xz - center));');
      sh.fragmentShader = 'varying float vFade;\n' + sh.fragmentShader.replace('#include <premultiplied_alpha_fragment>',
        'gl_FragColor.a *= vFade;\n#include <premultiplied_alpha_fragment>');
    };
    this.points = new THREE.Points(geo, mat);
    this.points.layers.set(LAYERS.over);
  }
  follow(x, y) {
    const s = this.spacing;
    this.points.position.set(Math.round(x / s) * s, Math.round(y / s) * s, 0);
    this.center.set(x, -y);   // MuJoCo (x, y) -> three world (x, -y) on the floor
  }
}

const EDGE_FRAG = /* glsl */ `
  uniform sampler2D tDepth; uniform vec2 texel; uniform float near, far, radius, thresh;
  uniform vec3 ink, bg; varying vec2 vUv;
  float lin(float d) { float z = d * 2.0 - 1.0; return 2.0 * near * far / (far + near - z * (far - near)); }
  void main() {
    float dc = lin(texture2D(tDepth, vUv).x), e = 0.0;
    for (int i = 0; i < 12; i++) {
      vec2 o = vec2(cos(float(i) * 0.5235988), sin(float(i) * 0.5235988)) * texel;
      e = max(e, (lin(texture2D(tDepth, vUv + o * radius).x) - dc) / dc);
      e = max(e, (lin(texture2D(tDepth, vUv + o * radius * 0.5).x) - dc) / dc);
    }
    gl_FragColor = vec4(mix(bg, ink, smoothstep(thresh * 0.5, thresh * 1.5, e)), 1.0);
    #include <colorspace_fragment>
  }`;

export class Stage {
  // floor: draw the dot floor; groundAtBottom: aim the camera so the floor plane meets the canvas
  // bottom edge (e.g. to stand the robot on a page rule drawn right below the canvas)
  constructor(canvas, { distance = 5.2, azimuth = 0.55, elevation = 0.1, height = 0.74, line = 1.4, floor = true, groundAtBottom = false } = {}) {
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, stencil: true }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.autoClear = false;
    this.line = line;
    this.scene = new THREE.Scene();
    this.world = new THREE.Group();          // MuJoCo z-up -> three y-up
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);
    this.camera = new THREE.PerspectiveCamera(20, 1, 1, 20);
    if (groundAtBottom) {                          // floor point under the target sits 5% above the bottom edge
      const t = Math.atan(0.95 * Math.tan((this.camera.fov / 2) * Math.PI / 180));
      height = distance * Math.cos(elevation) * Math.tan(elevation + t) - distance * Math.sin(elevation);
    }
    this.target = new THREE.Vector3(0, 0, height);   // MuJoCo coords
    Object.assign(this, { distance, azimuth, elevation });
    if (floor) { this.floor = new DotFloor(); this.world.add(this.floor.points); }

    this.rt = new THREE.WebGLRenderTarget(1, 1, { depthTexture: new THREE.DepthTexture(1, 1) });
    this.depthOnly = new THREE.MeshBasicMaterial({
      colorWrite: false, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      stencilWrite: true, stencilRef: 1, stencilFunc: THREE.AlwaysStencilFunc, stencilZPass: THREE.ReplaceStencilOp,
    });
    this.edge = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: this.rt.depthTexture }, texel: { value: new THREE.Vector2() },
        near: { value: 1 }, far: { value: 20 }, radius: { value: line }, thresh: { value: 0.012 },
        ink: { value: new THREE.Color(COLORS.ink) }, bg: { value: new THREE.Color(COLORS.bg) },
      },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: EDGE_FRAG, depthTest: false, depthWrite: false,
    });
    this.quadScene = new THREE.Scene();
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.edge));
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
    this.edge.uniforms.texel.value.set(1 / (w * pr), 1 / (h * pr));
    this.edge.uniforms.radius.value = this.line * pr;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // 0 = ink, 1 = accent: tints the contours (and optional extra line materials)
  setInk(k, ...lineMats) {
    const c = this.edge.uniforms.ink.value.set(COLORS.ink).lerp(new THREE.Color(COLORS.accent), k);
    for (const m of lineMats) m.color.copy(c);
  }

  follow(x, y, k = 0.08) {
    this.target.x += (x - this.target.x) * k;
    this.target.y += (y - this.target.y) * k;
    if (this.floor) this.floor.follow(this.target.x, this.target.y);
  }

  render() {
    const { target: t, azimuth: az, elevation: el, distance: d, renderer: r, camera: cam } = this;
    cam.position.set(t.x + d * Math.cos(el) * Math.cos(az), t.z + d * Math.sin(el), -(t.y + d * Math.cos(el) * Math.sin(az)));
    cam.lookAt(t.x, t.z, -t.y);
    cam.layers.set(LAYERS.fill);
    r.setRenderTarget(this.rt); r.setClearColor(COLORS.bg, 1); r.clear(); r.render(this.scene, cam);
    r.setRenderTarget(null); r.clear(); r.render(this.quadScene, this.quadCam);
    this.scene.overrideMaterial = this.depthOnly; r.render(this.scene, cam); this.scene.overrideMaterial = null;
    cam.layers.set(LAYERS.line); cam.layers.enable(LAYERS.over); cam.layers.enable(LAYERS.ghost);
    r.render(this.scene, cam);
  }
}
