// Earth-like atmosphere (Rayleigh + Mie + ozone) and the weather-map lookup shared by several shaders.
export const ATMO = /* glsl */`
const vec3 BETA_R = vec3(5.802e-6, 13.558e-6, 33.1e-6);
const float BETA_M = 1.2e-5;        // continental aerosol, AOD ~0.016: ~140 km visibility at sea level
const float BETA_M_EXT = 1.33e-5;
const vec3 BETA_O = vec3(0.650e-6, 1.881e-6, 0.085e-6);

vec3 atmoDensity(float h){
  return vec3(exp(-h/8000.0), exp(-h/1200.0), max(0.0, 1.0 - abs(h - 25000.0)/15000.0));
}
vec3 atmoExtinction(vec3 d){ return BETA_R*d.x + BETA_M_EXT*d.y + BETA_O*d.z; }

// Transmittance LUT: u = cos(sun zenith) warped around the horizon, v = sqrt(height fraction).
vec2 transUV(float h, float mu){
  return vec2(0.5 + 0.5*sign(mu)*sqrt(abs(mu)), sqrt(clamp(h/ATMO_H, 0.0, 1.0)));
}
vec3 sunTrans(sampler2D lut, float h, float mu){ return textureLod(lut, transUV(h, mu), 0.0).rgb; }

float phaseR(float c){ return 3.0/(16.0*PI)*(1.0 + c*c); }
float phaseHG(float c, float g){ float g2 = g*g; return (1.0 - g2)/(4.0*PI*pow(max(1.0 + g2 - 2.0*g*c, 1e-4), 1.5)); }
// Cornette-Shanks: HG with a Rayleigh-like back lobe, better for aerosols
float phaseCS(float c, float g){ return 1.5*(1.0 + c*c)/(2.0 + g*g)*phaseHG(c, g); }

// Equirectangular weather map: r = cloud cover, g = convective (storm) type, b = precipitation, a = lightning activity.
vec2 weatherUV(vec3 dir){
  return vec2(atan(dir.x, dir.z)*(0.5/PI) + 0.5, asin(clamp(dir.y, -1.0, 1.0))/PI + 0.5);
}
vec4 weatherAt(sampler2D map, vec3 dir){ return textureLod(map, weatherUV(dir), 0.0); }
// the same map through a cubic B-spline (4 bilinear taps): smooth, so thresholds show no texel staircase
vec4 weatherSmooth(sampler2D map, vec3 dir){
  vec2 size = vec2(textureSize(map, 0));
  vec2 x = weatherUV(dir)*size - 0.5, i = floor(x), f = x - i;
  vec2 w0 = (1.0 - f)*(1.0 - f)*(1.0 - f)/6.0, w1 = (4.0 - 6.0*f*f + 3.0*f*f*f)/6.0, w3 = f*f*f/6.0;
  vec2 g0 = w0 + w1, g1 = 1.0 - g0;
  vec2 h0 = (i - 0.5 + w1/g0)/size, h1 = (i + 1.5 + w3/g1)/size;
  return g0.y*(g0.x*textureLod(map, h0, 0.0) + g1.x*textureLod(map, vec2(h1.x, h0.y), 0.0))
       + g1.y*(g0.x*textureLod(map, vec2(h0.x, h1.y), 0.0) + g1.x*textureLod(map, h1, 0.0));
}
`
