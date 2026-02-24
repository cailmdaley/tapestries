// VellumShader.ts - WebGL shader for authentic portolan vellum background
// Ported from reference/combined-playground.html

import {
  ShaderMaterial,
  Mesh,
  PlaneGeometry,
  Color,
  DoubleSide,
} from 'three'

// Vertex shader - simple passthrough with UV coordinates
const vertexShader = `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

// Fragment shader - procedural vellum texture with organic variation
// Creates warm cream base with cloud-like beige variation, edge darkening,
// corner wear, and fine grain noise
const fragmentShader = `
  precision highp float;
  varying vec2 vUv;

  uniform vec3 uCenterColor;
  uniform float uWarmth;
  uniform float uEdgeDark;
  uniform float uCloudInt;
  uniform vec2 uAspect;

  // Simplex noise helpers
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                            + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m = m * m; m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  // Fractal brownian motion for organic cloud-like variation
  float fbm(vec2 p, float scale) {
    float f = 0.0;
    float amp = 0.5;
    vec2 pp = p * scale;
    for (int i = 0; i < 4; i++) {
      f += amp * snoise(pp);
      pp *= 2.0;
      amp *= 0.5;
    }
    return f;
  }

  // Simple hash for fine grain noise
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }

  void main() {
    vec2 uv = vUv;

    // Cloud-like organic variation (beige tones, not gray)
    float cloud = fbm(uv * uAspect, 3.0) * 0.5 + 0.5;
    float warmCloud = fbm(uv * uAspect + 10.0, 2.1) * 0.5 + 0.5;

    // Edge darkening - where hands would hold the vellum
    vec2 edgeDist = min(uv, 1.0 - uv);
    float edge = min(edgeDist.x, edgeDist.y);
    float wobble = snoise(uv * 8.0) * 0.02;
    edge += wobble;
    float edgeFade = smoothstep(0.0, 0.15, edge);
    float edgeEffect = (1.0 - edgeFade) * uEdgeDark;

    // Corner wear effect - more pronounced at corners
    float cornerDist = min(
      min(length(uv), length(uv - vec2(1.0, 0.0))),
      min(length(uv - vec2(0.0, 1.0)), length(uv - vec2(1.0, 1.0)))
    );
    float cornerEffect = exp(-cornerDist * 4.0) * 0.10;

    // Fine grain noise for paper texture
    float grain = (hash(uv * 1000.0) - 0.5) * 0.03;

    // Combine all effects
    float darkness =
      cloud * uCloudInt +
      warmCloud * uWarmth +
      edgeEffect +
      cornerEffect +
      grain;

    // Mix from center color to darker edge color
    vec3 edgeColor = uCenterColor * vec3(0.85, 0.80, 0.72);
    vec3 color = mix(uCenterColor, edgeColor, darkness);

    // Add warm tint to darker areas
    color.r += darkness * 0.03;
    color.g += darkness * 0.015;

    gl_FragColor = vec4(color, 1.0);
  }
`

// Default vellum parameters matching the combined playground "Authentic" preset
export interface VellumParams {
  centerColor: string  // Hex color for center (cleanest) area
  warmth: number       // Warm cloud intensity (0-0.3)
  edgeDark: number     // Edge darkening intensity (0-0.5)
  cloudInt: number     // Cool cloud intensity (0-0.3)
}

export const DEFAULT_VELLUM_PARAMS: VellumParams = {
  centerColor: '#f5eee1',  // Warm cream
  warmth: 0.12,
  edgeDark: 0.25,
  cloudInt: 0.12,
}

/**
 * Create a vellum ground plane with procedural shader
 * @param width World units width
 * @param height World units height
 * @param params Vellum appearance parameters
 */
export function createVellumPlane(
  width: number,
  height: number,
  params: VellumParams = DEFAULT_VELLUM_PARAMS
): Mesh {
  const geometry = new PlaneGeometry(width, height)

  // Parse hex color to RGB
  const color = new Color(params.centerColor)

  const material = new ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uCenterColor: { value: [color.r, color.g, color.b] },
      uWarmth: { value: params.warmth },
      uEdgeDark: { value: params.edgeDark },
      uCloudInt: { value: params.cloudInt },
      uAspect: { value: [width / height, 1.0] },
    },
    side: DoubleSide,
  })

  const mesh = new Mesh(geometry, material)
  mesh.rotation.x = -Math.PI / 2  // Lie flat on XZ plane
  mesh.receiveShadow = true

  return mesh
}

/**
 * Update vellum shader parameters at runtime
 */
export function updateVellumParams(mesh: Mesh, params: Partial<VellumParams>): void {
  const material = mesh.material as ShaderMaterial

  if (params.centerColor) {
    const color = new Color(params.centerColor)
    material.uniforms.uCenterColor.value = [color.r, color.g, color.b]
  }
  if (params.warmth !== undefined) {
    material.uniforms.uWarmth.value = params.warmth
  }
  if (params.edgeDark !== undefined) {
    material.uniforms.uEdgeDark.value = params.edgeDark
  }
  if (params.cloudInt !== undefined) {
    material.uniforms.uCloudInt.value = params.cloudInt
  }
}
