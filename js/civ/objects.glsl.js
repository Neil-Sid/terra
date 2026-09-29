// Shared shader pieces for 3D civilisation objects (buildings, vehicles, trains, people) so they are
// lit exactly like the terrain and depth-sort against it. Vertex shaders position everything relative
// to the camera (the camera sits at the origin) and write log depth like terrain.js.
import { GLSL_HEAD } from '../config.js'
import { NOISE } from '../glsl/noise.js'
import { TERRAIN } from '../glsl/terrain.glsl.js'
import { ATMO } from '../glsl/atmo.glsl.js'

// everything an object shader needs: constants, noise, terrain(p, maxF, info) with civ flattening, atmosphere
export const OBJ_COMMON = /* glsl */`${GLSL_HEAD}
${NOISE}
${TERRAIN}
${ATMO}
uniform vec3 uSun;
uniform float uTime, uLogDepthFC;
uniform sampler2D uTransLUT, uWeather;
`

// fragment-side lighting; up = local vertical (unit), h = height above sea level (m)
export const OBJ_LIGHT = /* glsl */`
float objNight(vec3 up){ return 1.0 - smoothstep(-0.1, 0.04, dot(up, uSun)); }
float objCloudShadow(vec3 up){
  float muS = dot(up, uSun);
  if (muS <= 0.0) return 1.0;
  vec3 q = normalize(up*(PLANET_R + 3000.0) + uSun*(3000.0/max(muS, 0.08)));
  return 1.0 - 0.8*smoothstep(0.25, 0.9, weatherAt(uWeather, q).r);
}
vec3 objSun(vec3 up, float h){ return SUN_I*sunTrans(uTransLUT, max(h, 0.0), dot(up, uSun))*objCloudShadow(up); }
vec3 objAmbient(vec3 up){ return SUN_I*vec3(0.030, 0.045, 0.075)*smoothstep(-0.18, 0.35, dot(up, uSun)); }
// diffuse albedo lit by sun + sky; add emission separately
vec3 objLight(vec3 albedo, vec3 N, vec3 up, float h){
  return albedo*(objSun(up, h)*max(dot(N, uSun), 0.0) + objAmbient(up)*(0.6 + 0.4*dot(N, up)));
}
float objSpec(vec3 N, vec3 V, float rough){
  vec3 H = normalize(V + uSun);
  float a2 = rough*rough*rough*rough, nh = max(dot(N, H), 0.0), d = nh*nh*(a2 - 1.0) + 1.0;
  float F = 0.04 + 0.96*pow(1.0 - clamp(dot(V, H), 0.0, 1.0), 5.0);
  return a2/(PI*d*d)*F*0.25*max(dot(N, uSun), 0.0);
}
`

// uniforms every object material must supply
export function objUniforms(shared, FAR) {
  return {
    uSun: shared.uSun, uTime: shared.uTime, uTransLUT: shared.uTransLUT, uWeather: shared.uWeather,
    uLogDepthFC: { value: 2.0 / Math.log2(FAR + 1) }, ...shared.civ,
  }
}
