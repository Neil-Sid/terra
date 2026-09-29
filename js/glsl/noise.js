// Simplex noise (Ashima/Gustavson) with analytic gradient, plus small hashes.
export const NOISE = /* glsl */`
// Always 0 (never set). D3D inlines every call site and unrolls constant-bound loops, so each snoised() call costs
// seconds of shader compile: batch evaluations through one call site in a loop bounded by loopN(n), which it can't unroll.
uniform float uLoop0;
int loopN(int n){ return n + int(uLoop0); }

vec3 mod289(vec3 x){ return x - floor(x*(1.0/289.0))*289.0; }
vec4 mod289(vec4 x){ return x - floor(x*(1.0/289.0))*289.0; }
vec4 permute(vec4 x){ return mod289(((x*34.0) + 10.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314*r; }

// snoised()/105. Batched callers take this and multiply by 105 at each use, the way the expression reads with snoised():
// fxc folds that constant into the use (e.g. 105*0.65), so this keeps their results bit-identical to separate snoised() calls.
vec4 snoisedR(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - 0.5;
  i = mod289(i);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  vec4 j = p - 49.0*floor(p*(1.0/49.0));
  vec4 x_ = floor(j*(1.0/7.0));
  vec4 y_ = floor(j - 7.0*x_);
  vec4 x = (x_*2.0 + 0.5)/7.0 - 1.0;
  vec4 y = (y_*2.0 + 0.5)/7.0 - 1.0;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 g0 = vec3(a0.xy, h.x);
  vec3 g1 = vec3(a0.zw, h.y);
  vec3 g2 = vec3(a1.xy, h.z);
  vec3 g3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(g0,g0), dot(g1,g1), dot(g2,g2), dot(g3,g3)));
  g0 *= norm.x; g1 *= norm.y; g2 *= norm.z; g3 *= norm.w;
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  vec4 m2 = m*m, m3 = m2*m, m4 = m2*m2;
  vec4 pdx = vec4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
  vec3 grad = -8.0*(m3.x*pdx.x*x0 + m3.y*pdx.y*x1 + m3.z*pdx.z*x2 + m3.w*pdx.w*x3)
            + m4.x*g0 + m4.y*g1 + m4.z*g2 + m4.w*g3;
  return vec4(dot(m4, pdx), grad);
}
// Returns (value, d/dx, d/dy, d/dz); value is roughly in [-1, 1].
vec4 snoised(vec3 v){ return 105.0*snoisedR(v); }
float snoise(vec3 v){ return snoised(v).x; }

float hash13(vec3 p){
  p = fract(p*0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y)*p.z);
}
float hash12(vec2 p){
  vec3 p3 = fract(vec3(p.xyx)*0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y)*p3.z);
}
`
