#!/usr/bin/env python3
"""Ray-trace a Schwarzschild black hole with a thin accretion disk.

Every pixel is a photon traced backwards from the camera through curved
spacetime. In units where the Schwarzschild radius r_s = 1, a null geodesic
obeys

    d2x/dl2 = -(3/2) * h^2 * x / |x|^5        with  h = |x cross v|  (conserved)

which is integrated with RK4. Rays that fall below r = 1 are the shadow, rays
that cross the equatorial plane pick up disk emission (relativistic Doppler
beaming + gravitational redshift), rays that escape sample a procedural sky.

Outputs
    <out>.jpg        tone-mapped image used as the banner backdrop
    <out>.png        lossless copy for inspection
    <out>_maps.npz   first plane-crossing radius / azimuth / redshift per
                     pixel, used to animate physically-correct orbiting
                     hot spots in the SVG

Usage
    python scripts/render_blackhole.py --scale 1.5 --ss 2 --out build/blackhole
"""
import argparse
import math
import os

import numpy as np
from numba import njit, prange
from PIL import Image
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------- scene ----
VIEW_W, VIEW_H = 1600, 640        # SVG viewBox the image is composed for
HOLE_X, HOLE_Y = 1204.0, 316.0    # where the singularity sits in the viewBox
PX_PER_RS = 31.0                  # viewBox pixels per Schwarzschild radius
CAM_DIST = 42.0                   # camera distance, in r_s
CAM_ELEV = math.radians(8.5)      # camera elevation above the disk plane
CAM_ROLL = math.radians(-9.0)     # roll of the camera around its view axis
R_IN, R_OUT = 3.0, 13.5           # disk inner edge (ISCO) and outer edge
R_ESC = 70.0                      # rays past this radius have escaped
T_IN = 6300.0                     # colour temperature at the inner edge (K)
DOPPLER_POWER = 2.6               # beaming exponent (4 = bolometric, harsh)

rng = np.random.default_rng(7)
NOISE2 = rng.random((256, 256))
PERM = np.concatenate([rng.permutation(256)] * 2).astype(np.int64)


# ---------------------------------------------------------------- noise ----
@njit(cache=True, fastmath=True)
def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True)
def vnoise2(x, y, tab):
    ix = math.floor(x)
    iy = math.floor(y)
    fx = _smooth(x - ix)
    fy = _smooth(y - iy)
    i0 = int(ix) & 255
    j0 = int(iy) & 255
    i1 = (i0 + 1) & 255
    j1 = (j0 + 1) & 255
    a = tab[i0, j0] + (tab[i1, j0] - tab[i0, j0]) * fx
    b = tab[i0, j1] + (tab[i1, j1] - tab[i0, j1]) * fx
    return a + (b - a) * fy


@njit(cache=True, fastmath=True)
def fbm2(x, y, tab, octaves):
    s = 0.0
    amp = 0.5
    norm = 0.0
    for _ in range(octaves):
        s += amp * vnoise2(x, y, tab)
        norm += amp
        x = x * 2.03 + 17.1
        y = y * 2.03 + 9.7
        amp *= 0.5
    return s / norm


@njit(cache=True, fastmath=True)
def vnoise3(x, y, z, perm):
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = _smooth(x - ix)
    fy = _smooth(y - iy)
    fz = _smooth(z - iz)
    X = int(ix) & 255
    Y = int(iy) & 255
    Z = int(iz) & 255
    out = 0.0
    for dz in range(2):
        wz = fz if dz else 1.0 - fz
        for dy in range(2):
            wy = fy if dy else 1.0 - fy
            for dx in range(2):
                wx = fx if dx else 1.0 - fx
                h = perm[(perm[(perm[(X + dx) & 255] + Y + dy) & 255] + Z + dz) & 255]
                out += wx * wy * wz * (h / 255.0)
    return out


@njit(cache=True, fastmath=True)
def fbm3(x, y, z, perm, octaves):
    s = 0.0
    amp = 0.5
    norm = 0.0
    for _ in range(octaves):
        s += amp * vnoise3(x, y, z, perm)
        norm += amp
        x = x * 2.01 + 3.3
        y = y * 2.01 + 7.7
        z = z * 2.01 + 1.9
        amp *= 0.5
    return s / norm


@njit(cache=True)
def hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h ^ (h >> 16)) & 0xFFFFFFFF


@njit(cache=True)
def lcg(h):
    return (h * 1664525 + 1013904223) & 0xFFFFFFFF


# --------------------------------------------------------------- colour ----
@njit(cache=True, fastmath=True)
def blackbody(T):
    """Approximate blackbody chromaticity in linear RGB, peak channel = 1."""
    t = min(max(T, 1000.0), 40000.0) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19.0 else 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60.0) ** -0.1332047592
        g = 288.1221695283 * (t - 60.0) ** -0.0755148492
        b = 255.0
    r = min(max(r, 0.0), 255.0) / 255.0
    g = min(max(g, 0.0), 255.0) / 255.0
    b = min(max(b, 0.0), 255.0) / 255.0
    return r ** 2.2, g ** 2.2, b ** 2.2


# ------------------------------------------------------------------ sky ----
@njit(cache=True, fastmath=True)
def sky(dx, dy, dz, perm, star_sigma):
    # faint galactic band + dust
    bx, by, bz = 0.884, -0.069, -0.464        # band normal
    lat = dx * bx + dy * by + dz * bz - 0.13
    band = math.exp(-(lat * lat) / 0.016)
    n1 = fbm3(dx * 4.2 + 11.0, dy * 4.2, dz * 4.2, perm, 5)
    n2 = fbm3(dx * 11.0, dy * 11.0 + 5.0, dz * 11.0, perm, 4)
    lanes = min(max((n1 - 0.38) / 0.30, 0.0), 1.0)
    glow = band * _smooth(lanes) * (0.35 + 1.3 * n2)
    r = 0.014 * glow
    g = 0.021 * glow
    b = 0.050 * glow
    # warm core of the band
    core = math.exp(-(lat * lat) / 0.003) * max(0.0, n2 - 0.42) * _smooth(lanes)
    r += 0.030 * core
    g += 0.017 * core
    b += 0.012 * core

    # hashed stars on a thin spherical shell of lattice cells
    S = 85.0
    px, py, pz = dx * S, dy * S, dz * S
    cx, cy, cz = int(math.floor(px)), int(math.floor(py)), int(math.floor(pz))
    density = 0.085 + 0.40 * band
    inv2s2 = 1.0 / (2.0 * star_sigma * star_sigma)
    for oz in range(-1, 2):
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                ix, iy, iz = cx + ox, cy + oy, cz + oz
                h = hash3(ix + 4096, iy + 4096, iz + 4096)
                if (h & 0xFFFF) / 65535.0 > density:
                    continue
                h = lcg(h)
                qx = ix + (h & 0xFFFF) / 65535.0
                h = lcg(h)
                qy = iy + (h & 0xFFFF) / 65535.0
                h = lcg(h)
                qz = iz + (h & 0xFFFF) / 65535.0
                q = math.sqrt(qx * qx + qy * qy + qz * qz)
                if q < S - 0.5 or q > S + 0.5:
                    continue
                ex, ey, ez = qx / q - dx, qy / q - dy, qz / q - dz
                d2 = ex * ex + ey * ey + ez * ez
                w = d2 * inv2s2
                if w > 14.0:
                    continue
                h = lcg(h)
                u = ((h >> 8) & 0xFFFF) / 65535.0
                mag = 0.011 * (u + 0.003) ** -1.3       # power-law brightness
                h = lcg(h)
                tcol = 3600.0 + 9500.0 * (((h >> 8) & 0xFFFF) / 65535.0) ** 0.7
                sr, sg, sb = blackbody(tcol)
                e = mag * math.exp(-w)
                r += e * sr
                g += e * sg
                b += e * sb
    return r, g, b


# ----------------------------------------------------------------- disk ----
@njit(cache=True, fastmath=True)
def doppler(rc, phi, vx, vy, vz):
    """Redshift factor g = E_obs / E_emit for Keplerian matter at (rc, phi).

    (vx, vy, vz) is the backwards-traced ray direction at the crossing.
    """
    vk = math.sqrt(0.5 / (rc - 1.0))
    vn = math.sqrt(vx * vx + vy * vy + vz * vz)
    # photon travels along -v; orbital direction is (-sin phi, cos phi, 0)
    cosa = (math.sin(phi) * vx - math.cos(phi) * vy) / vn
    gamma = 1.0 / math.sqrt(1.0 - vk * vk)
    return math.sqrt(1.0 - 1.0 / rc) / (gamma * (1.0 - vk * cosa))


@njit(cache=True, fastmath=True)
def disk(rc, phi, vx, vy, vz, tab):
    """Emission (linear RGB) and opacity of the disk at radius rc, azimuth phi."""
    e_in = min(max((rc - R_IN) / 0.45, 0.0), 1.0)
    e_out = 1.0 - min(max((rc - R_OUT * 0.52) / (R_OUT * 0.48), 0.0), 1.0)
    env = _smooth(e_in) * _smooth(e_out) ** 1.6
    if env <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    # turbulence wound up by differential (Keplerian) rotation
    wind = phi + 46.0 * rc ** -1.5
    u = rc * math.cos(wind)
    v = rc * math.sin(wind)
    streaks = fbm2(u * 0.9 + 40.0, v * 0.9 + 40.0, tab, 5)
    fine = fbm2(u * 3.1 + 80.0, v * 3.1 + 15.0, tab, 3)
    bands = fbm2(rc * 2.3 + 3.0, 0.5, tab, 4)
    tex = (0.25 + 1.5 * streaks) ** 2.0 * (0.45 + 1.1 * bands) * (0.7 + 0.6 * fine)

    g = doppler(rc, phi, vx, vy, vz)
    temp = T_IN * (rc / R_IN) ** -0.75 * g
    cr, cg, cb = blackbody(temp)
    inten = 2.6 * (R_IN / rc) ** 2.1 * g ** DOPPLER_POWER * tex * env
    alpha = env * min(0.96, 0.5 + 0.55 * tex)
    return inten * cr, inten * cg, inten * cb, alpha


# ------------------------------------------------------------ geodesics ----
@njit(cache=True, fastmath=True)
def trace(ox, oy, oz, dx, dy, dz, tab, perm, star_sigma, want_map):
    """Trace one ray. Returns colour plus first plane-crossing data."""
    x, y, z = ox, oy, oz
    vx, vy, vz = dx, dy, dz
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    k = -1.5 * (hx * hx + hy * hy + hz * hz)

    cr = cg = cb = 0.0
    trans = 1.0
    ncross = 0
    m_r = -1.0
    m_phi = 0.0
    m_g = 0.0
    captured = False

    for _ in range(4000):
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        if r < 1.0:
            captured = True
            break
        if r > R_ESC and (x * vx + y * vy + z * vz) > 0.0:
            break
        dt = min(max(0.028 * r, 0.02), 1.2)

        # RK4 on (x, v)
        a = k / (r2 * r2 * r)
        k1x, k1y, k1z = a * x, a * y, a * z
        x2, y2, z2 = x + 0.5 * dt * vx, y + 0.5 * dt * vy, z + 0.5 * dt * vz
        u2x, u2y, u2z = vx + 0.5 * dt * k1x, vy + 0.5 * dt * k1y, vz + 0.5 * dt * k1z
        q = x2 * x2 + y2 * y2 + z2 * z2
        a = k / (q * q * math.sqrt(q))
        k2x, k2y, k2z = a * x2, a * y2, a * z2
        x3, y3, z3 = x + 0.5 * dt * u2x, y + 0.5 * dt * u2y, z + 0.5 * dt * u2z
        u3x, u3y, u3z = vx + 0.5 * dt * k2x, vy + 0.5 * dt * k2y, vz + 0.5 * dt * k2z
        q = x3 * x3 + y3 * y3 + z3 * z3
        a = k / (q * q * math.sqrt(q))
        k3x, k3y, k3z = a * x3, a * y3, a * z3
        x4, y4, z4 = x + dt * u3x, y + dt * u3y, z + dt * u3z
        u4x, u4y, u4z = vx + dt * k3x, vy + dt * k3y, vz + dt * k3z
        q = x4 * x4 + y4 * y4 + z4 * z4
        a = k / (q * q * math.sqrt(q))
        k4x, k4y, k4z = a * x4, a * y4, a * z4

        nx = x + dt / 6.0 * (vx + 2.0 * u2x + 2.0 * u3x + u4x)
        ny = y + dt / 6.0 * (vy + 2.0 * u2y + 2.0 * u3y + u4y)
        nz = z + dt / 6.0 * (vz + 2.0 * u2z + 2.0 * u3z + u4z)
        nvx = vx + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        nvy = vy + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        nvz = vz + dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

        if (z > 0.0) != (nz > 0.0):                 # crossed the disk plane
            t = z / (z - nz)
            px = x + t * (nx - x)
            py = y + t * (ny - y)
            rc = math.sqrt(px * px + py * py)
            if rc > 1.02:
                phi = math.atan2(py, px)
                if want_map and ncross == 0:
                    m_r = rc
                    m_phi = phi
                    m_g = doppler(rc, phi, vx, vy, vz) if rc > 1.6 else 0.0
                if R_IN <= rc <= R_OUT and trans > 0.003:
                    er, eg, eb, al = disk(rc, phi, vx, vy, vz, tab)
                    cr += trans * er
                    cg += trans * eg
                    cb += trans * eb
                    trans *= 1.0 - al
            ncross += 1

        x, y, z = nx, ny, nz
        vx, vy, vz = nvx, nvy, nvz

    if not captured and trans > 0.003:
        vn = math.sqrt(vx * vx + vy * vy + vz * vz)
        sr, sg, sb = sky(vx / vn, vy / vn, vz / vn, perm, star_sigma)
        cr += trans * sr
        cg += trans * sg
        cb += trans * sb
    return cr, cg, cb, m_r, m_phi, m_g


@njit(parallel=True, cache=True, fastmath=True)
def render(w, h, ss, cam, fwd, right, up, focal, cx, cy, tab, perm, star_sigma):
    img = np.zeros((h, w, 3), np.float32)
    inv = 1.0 / (ss * ss)
    for j in prange(h):
        for i in range(w):
            ar = ag = ab = 0.0
            for sj in range(ss):
                for si in range(ss):
                    X = i + (si + 0.5) / ss - cx
                    Y = cy - (j + (sj + 0.5) / ss)
                    dx = fwd[0] * focal + right[0] * X + up[0] * Y
                    dy = fwd[1] * focal + right[1] * X + up[1] * Y
                    dz = fwd[2] * focal + right[2] * X + up[2] * Y
                    n = math.sqrt(dx * dx + dy * dy + dz * dz)
                    r, g, b, _, _, _ = trace(cam[0], cam[1], cam[2], dx / n, dy / n, dz / n,
                                             tab, perm, star_sigma, False)
                    ar += r
                    ag += g
                    ab += b
            img[j, i, 0] = ar * inv
            img[j, i, 1] = ag * inv
            img[j, i, 2] = ab * inv
    return img


@njit(parallel=True, cache=True, fastmath=True)
def render_maps(w, h, cam, fwd, right, up, focal, cx, cy, tab, perm):
    maps = np.zeros((h, w, 3), np.float32)
    for j in prange(h):
        for i in range(w):
            X = i + 0.5 - cx
            Y = cy - (j + 0.5)
            dx = fwd[0] * focal + right[0] * X + up[0] * Y
            dy = fwd[1] * focal + right[1] * X + up[1] * Y
            dz = fwd[2] * focal + right[2] * X + up[2] * Y
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            _, _, _, mr, mp, mg = trace(cam[0], cam[1], cam[2], dx / n, dy / n, dz / n,
                                        tab, perm, 1e-3, True)
            maps[j, i, 0] = mr
            maps[j, i, 1] = mp
            maps[j, i, 2] = mg
    return maps


# ------------------------------------------------------------- pipeline ----
def camera():
    cam = CAM_DIST * np.array([0.0, -math.cos(CAM_ELEV), math.sin(CAM_ELEV)])
    fwd = -cam / np.linalg.norm(cam)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c, s = math.cos(CAM_ROLL), math.sin(CAM_ROLL)
    return cam, fwd, c * right + s * up, -s * right + c * up


def tonemap(hdr, scale):
    """Bloom, filmic curve, vignette and grain."""
    bloom = np.zeros_like(hdr)
    for sigma, gain in ((2.5, 0.20), (9.0, 0.16), (30.0, 0.13), (90.0, 0.09)):
        for c in range(3):
            bloom[..., c] += gain * gaussian_filter(hdr[..., c], sigma * scale)
    x = hdr + bloom
    h, w, _ = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - HOLE_X * scale) / w, (yy - HOLE_Y * scale) / w)
    x *= (1.0 - 0.55 * np.clip(d - 0.15, 0, 1) ** 1.4)[..., None]
    x = 1.0 - np.exp(-1.25 * x)                       # soft shoulder
    x = np.clip(x, 0, 1) ** (1 / 2.2)
    x = x + (x * x * (3 - 2 * x) - x) * 0.25          # gentle contrast
    x += np.random.default_rng(3).normal(0, 0.006, x.shape[:2])[..., None]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.5, help="output px per viewBox px")
    ap.add_argument("--ss", type=int, default=2, help="supersampling per axis")
    ap.add_argument("--out", default="build/blackhole")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--maps-only", action="store_true", help="skip the image, rewrite the maps")
    args = ap.parse_args()

    s = args.scale
    w, h = round(VIEW_W * s), round(VIEW_H * s)
    cam, fwd, right, up = camera()
    focal = PX_PER_RS * CAM_DIST * s
    cx, cy = HOLE_X * s, HOLE_Y * s
    star_sigma = 0.62 / focal

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not args.maps_only:
        hdr = render(w, h, args.ss, cam, fwd, right, up, focal, cx, cy, NOISE2, PERM, star_sigma)
        img = Image.fromarray(tonemap(hdr, s))
        img.save(args.out + ".png")
        img.save(args.out + ".jpg", quality=args.quality, subsampling=0, optimize=True, progressive=True)

    # maps are always rendered at viewBox resolution x2
    ms = 2.0
    maps = render_maps(round(VIEW_W * ms), round(VIEW_H * ms), cam, fwd, right, up,
                       PX_PER_RS * CAM_DIST * ms, HOLE_X * ms, HOLE_Y * ms, NOISE2, PERM)
    np.savez_compressed(args.out + "_maps.npz", maps=maps, scale=ms)
    print(f"wrote {args.out}  {w}x{h}  ss={args.ss}")


if __name__ == "__main__":
    main()
