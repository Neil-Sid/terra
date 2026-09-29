// Planet-wide constants. Units are metres and seconds everywhere.
export const R = 6371000            // sea-level radius
export const ATMO_H = 100000        // top of the atmosphere
export const CLOUD_BASE = 1400
export const CLOUD_TOP = 9500
export const GRID = 64              // quads per terrain patch edge
export const MAX_LEVEL = 19         // deepest quadtree level (~0.3 m vertex spacing)
export const FAR = 2e8
export const SUN_I = 10.0           // sun irradiance in scene units

// Same constants for shaders.
export const GLSL_HEAD = /* glsl */`
precision highp float;
precision highp int;
precision highp sampler3D;
#define PI 3.14159265359
#define PLANET_R ${R.toFixed(1)}
#define ATMO_H ${ATMO_H.toFixed(1)}
#define CLOUD_BASE ${CLOUD_BASE.toFixed(1)}
#define CLOUD_TOP ${CLOUD_TOP.toFixed(1)}
#define GRID ${GRID.toFixed(1)}
#define SUN_I ${SUN_I.toFixed(2)}
`
