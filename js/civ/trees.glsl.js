// Shaders for the 3D trees (trees.js). A placement pass evaluates terrainCiv() once per candidate (the only terrain()
// call site): ground height, forestDensity() with clearings, farm fields and water, then kind, size and tint. The tree
// meshes read that texel by gl_InstanceID and collapse instances with no tree. Notes: notes/forests.md.
import { GLSL_HEAD } from '../config.js'
import { NOISE } from '../glsl/noise.js'
import { TERRAIN } from '../glsl/terrain.glsl.js'
import { FOREST } from '../glsl/forest.glsl.js'
import { FARM_PAINT } from './farm.glsl.js'
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

export const PLACE_W = 256              // candidates per row of the placement textures

// per candidate: x = ground height, y = kind + 0.9*autumn (-1 = no tree), z = height (m), w = dry tone 0..1
// kinds: 0 conifer, 1 broadleaf, 2 palm, 3 shrub, 4 acacia (3 and 4 use the broadleaf mesh)
export const PLACE_FRAG = /* glsl */`${GLSL_HEAD}
uniform highp sampler2D uCand;          // xyz = direction, w = random 0..1
uniform int uCount;
uniform vec3 uSun;                      // declared for forestPaint(), unused here
uniform float uPixAngle;
out vec4 outP;
${NOISE}
${TERRAIN}
float civBand(float d, float w, float aa){ return clamp((w - d)/aa + 0.5, 0.0, 1.0)*min(1.0, 2.0*w/aa + 0.02); }
float civCanopy;                        // written by forestPaint() for civScatter() (paint.glsl.js); unused here
${FARM_PAINT}
${FOREST}
void main(){
  ivec2 px = ivec2(gl_FragCoord.xy);
  if (px.y*${PLACE_W} + px.x >= uCount){ outP = vec4(0.0, -1.0, 0.0, 0.0); return; }
  vec4 cd = texelFetch(uCand, px, 0);
  vec3 p = normalize(cd.xyz), info;
  Civ c;
  vec4 t = terrainCiv(p, 1.2e6, info, c);
  float h = t.x;
  vec3 g = t.yzw - p*dot(t.yzw, p);
  float slope = 1.0 - inversesqrt(1.0 + dot(g, g)/(PLANET_R*PLANET_R));
  vec2 cl = climate(p, h, info.x);
  float T = cl.x, M = cl.y;
  vec4 fc = forestClimate(T, M, h);
  float dens = forestDensity(p, T, M, slope, h, c)*forestClear(c);
  // farm fields (the farm map at full resolution: no derivatives in this pass)
  vec2 uv = vec2(atan(p.x, p.z)*(0.5/PI) + 0.5, asin(clamp(p.y, -1.0, 1.0))/PI + 0.5);
  farmTerr = textureLod(uFarmTerr, uv, 0.0).rg*vec2(1.0, 1.5);
  vec2 fw = farmFieldWeight(p, textureLod(uFarmMap, uv, 0.0).r, T, M, slope, h, c);
  bool hedge = fw.y > 0.0 && cd.w < 0.4*smoothstep(0.0, 0.2, fc.x);
  dens *= 1.0 - fw.x;
  // never in the water: channel (plus the bank), lakes, the sea
  bool wet = h < max(0.0, c.lakeLevel) + 0.3 || h < c.water - 0.05 || c.dRiver < c.riverHalf + 0.4*forestBank(c);
  if (wet || !hedge && cd.w >= dens){ outP = vec4(h, -1.0, 0.0, 0.0); return; }
  float r2 = fract(cd.w*97.13 + 0.31), r3 = fract(cd.w*531.7 + 0.77);
  float open = smoothstep(0.7, 0.3, fc.y), vig = smoothstep(-3.0, 8.0, T)*smoothstep(0.25, 0.6, M);
  float palm = fc.w*(0.75*smoothstep(40.0, 8.0, h) + 0.08)*(1.0 - open);
  int kind = 1;
  float H;
  if (hedge){ kind = 1; H = 7.0 + 7.0*r3; }
  else if (r2 < fc.z){ kind = 0; H = mix(9.0, 24.0, vig)*(0.7 + 0.5*r3); }
  else if (r2 < fc.z + palm){ kind = 2; H = 9.0 + 11.0*r3; }
  else if (open > 0.5 && fc.w > 0.5 && r3 < 0.7){ kind = 4; H = 6.0 + 5.0*r3; }
  else if (r3 < 0.1 + 0.5*open){ kind = 3; H = 1.5 + 2.5*r2; }
  else H = mix(10.0, mix(22.0, 34.0, fc.w), vig)*(0.7 + 0.5*r3);
  H *= 0.4 + 0.6*smoothstep(-4.5, 1.0, T);                          // stunted near the treeline
  float season = fract(uFarmDay/365.0 + (p.y < 0.0 ? 0.5 : 0.0));
  float autumn = smoothstep(0.72, 0.79, season)*(1.0 - smoothstep(0.84, 0.88, season))*(1.0 - fc.w)*smoothstep(17.0, 9.0, T);
  outP = vec4(h, float(kind) + 0.9*(kind == 1 ? autumn : 0.0), H, open);
}`

// compact = instances carry their placement (iPlace, one kind per mesh, from a readback); else texelFetch it
export const treeVert = compact => /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform mat4 projectionMatrix, viewMatrix;
uniform highp sampler2D uPlace;
uniform vec3 uAnchor;                   // rebuild anchor minus camera (m)
uniform int uKind;                      // this mesh: 0 conifer, 1 broadleaf (shrubs, acacias), 2 palm
uniform float uTreeR0, uTreeMax;
in vec3 position, normal;               // unit tree, z up, height 1
in vec2 part;                           // 0 trunk / 1 foliage, height in the tree 0..1
in vec3 iPos, iUp;                      // sea-level anchor relative to the rebuild anchor, local vertical
in vec2 iMisc;                          // rank (drawn while d < r0/sqrt(rank)), random 0..1
${compact ? 'in vec4 iPlace;' : ''}
out vec3 vN, vRel, vLoc;
out float vFragDepth, vAO, vTier;
flat out vec3 vUp, vAlb, vSunC, vAmbC;
flat out float vFade, vFol;
void main(){
  vec4 pl = ${compact ? 'iPlace' : `texelFetch(uPlace, ivec2(gl_InstanceID % ${PLACE_W}, gl_InstanceID / ${PLACE_W}), 0)`};
  int kind = int(floor(pl.y));
  vec3 base = uAnchor + iPos + iUp*pl.x;
  float d = length(base), dvis = min(uTreeR0*inversesqrt(max(iMisc.x, 1e-6)), uTreeMax);
  vFade = min(clamp((dvis - d)/(0.25*dvis), 0.0, 1.0), clamp((uTreeMax - d)/(0.5*uTreeMax), 0.0, 1.0));   // long dither to the far end
  float rnd = iMisc.y, H = pl.z, a = rnd*37.7;
  // no tree, another mesh's kind, faded out or outside the view: collapse
  vec4 cc = projectionMatrix*viewMatrix*vec4(base + iUp*(0.5*H), 1.0);
  float rc = 0.7*H*(max(projectionMatrix[0][0], projectionMatrix[1][1]) + 1.0);
  if (pl.y < 0.0 || (kind > 2 ? 1 : kind) != uKind || vFade <= 0.0 || cc.w < -H || abs(cc.x) > cc.w + rc || abs(cc.y) > cc.w + rc){
    gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
    return;
  }
  vec3 U = iUp, E0 = normalize(cross(vec3(0.0, 1.0, 0.0), U)), N0 = cross(U, E0);
  vec3 E = E0*cos(a) + N0*sin(a), Nn = cross(U, E);
  vec3 lp = position, nl = normal;
  float wide = 0.8 + 0.4*fract(rnd*13.1);
  if (part.x > 0.5 && uKind == 1){
    // lumpy crowns; acacias flatten into umbrellas, shrubs sit on the ground
    float lump = 1.0 + 0.22*snoise(position*5.0 + rnd*31.0);
    vec3 c = vec3(0.0, 0.0, 0.62), o = (position - c)*lump;
    if (kind == 4){ o *= vec3(1.5, 1.5, 0.35); c.z = 0.82; }
    if (kind == 3){ o *= vec3(1.3, 1.3, 1.1); c.z = 0.4; }
    lp = c + o;
    nl = normalize(nl/vec3(kind == 4 ? 1.5 : 1.0, kind == 4 ? 1.5 : 1.0, kind == 4 ? 0.35 : 1.0));
  }
  if (part.x < 0.5 && kind == 3) lp.z -= 1.0;                       // shrubs: no trunk
  lp.xy *= wide;
  // sway: the crown bends with a slow gust
  float sw = part.y*part.y*H*0.012*(sin(uTime*1.1 + rnd*40.0) + 0.5*sin(uTime*2.3 + rnd*17.0));
  vec3 rel = base + (E*lp.x + Nn*lp.y)*H + U*(lp.z*H) + E*sw;
  vN = normalize(E*nl.x + Nn*nl.y + U*nl.z);
  vRel = rel;
  vUp = U;
  vLoc = lp*H + rnd*vec3(71.0, 37.0, 53.0);                          // tree-local metres (texture that doesn't swim)
  vFol = part.x;
  // conifer tiers: 0 at a tier's apex (in the shade of the tier above) .. 1 at its rim; the top tier is all lit
  vTier = uKind == 0 && part.x > 0.5 && nl.z > 0.99 && position.z < 0.99 ? 0.0 : 1.0;
  vAO = part.x > 0.5 ? mix(0.4, 1.0, clamp(part.y*1.3 - 0.1, 0.0, 1.0))*(nl.z < -0.5 ? 0.6 : 1.0) : 0.7;
  // colour per tree: kind, tone, autumn tint
  float tone = 0.8 + 0.4*fract(rnd*71.3), autumn = kind == 1 ? fract(pl.y)/0.9 : 0.0;
  vec3 leaf = kind == 0 ? vec3(0.040, 0.072, 0.026) : kind == 2 ? vec3(0.07, 0.10, 0.035) : vec3(0.045, 0.085, 0.030);
  leaf = mix(leaf, vec3(0.075, 0.08, 0.04), pl.w*(kind == 0 ? 0.3 : 0.8));
  leaf = mix(leaf, mix(vec3(0.20, 0.12, 0.02), vec3(0.17, 0.05, 0.02), fract(rnd*5.7)), autumn*step(0.3, fract(rnd*3.3)));
  vAlb = part.x > 0.5 ? leaf*tone : vec3(0.10, 0.074, 0.052)*tone;                // grey-brown bark
  vSunC = objSun(U, 200.0)*(part.x > 0.5 ? mix(0.5, 1.0, part.y) : 0.4);      // neighbours shade trunks and lower crowns
  vAmbC = objAmbient(U);
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
  vFragDepth = 1.0 + gl_Position.w;
}`

export const TREE_FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
in vec3 vN, vRel, vLoc;
in float vFragDepth, vAO, vTier;
flat in vec3 vUp, vAlb, vSunC, vAmbC;
flat in float vFade, vFol;
out vec4 outColor;
void main(){
  if (vFade < 1.0 && hash12(gl_FragCoord.xy) >= vFade) discard;
  vec3 N = normalize(vN);
  if (!gl_FrontFacing) N = -N;
  float ndl = dot(N, uSun);
  // needle / leaf clumps (~0.3 m and ~0.12 m) or bark furrows, faded out below ~2 px; conifer tiers shade the one below
  float px = length(fwidth(vLoc));
  vec3 x = vFol > 0.5 ? vLoc : vLoc*vec3(1.0, 1.0, 0.2);
  float tex = snoise(x*3.0)*(1.0 - smoothstep(0.08, 0.2, px)) + 0.5*snoise(x*8.0 + 3.1)*(1.0 - smoothstep(0.03, 0.08, px));
  vec3 alb = vAlb*(1.0 + 0.28*tex)*mix(0.55, 1.0, smoothstep(0.35, 1.0, vTier));
  // foliage: wrapped light (leaves scatter), darker deep in the crown
  vec3 col = alb*(vSunC*clamp(ndl*0.8 + 0.15, 0.0, 1.0) + vAmbC*(0.75 + 0.25*dot(N, vUp)))*vAO;
  outColor = vec4(col, 1.0);
  gl_FragDepth = log2(vFragDepth)*uLogDepthFC*0.5;
}`
