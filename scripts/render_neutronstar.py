#!/usr/bin/env python3
"""Ray-trace a pulsar: a neutron star seen through its own curved spacetime.

Units: Schwarzschild radius r_s = 1. The star has radius R = 2.5 r_s
(about 10 km for 1.4 solar masses), which is outside the photon sphere
(1.5 r_s), so no light is captured - but it is bent hard enough that the
camera sees roughly 80 % of the surface at once, both magnetic poles included.

Every pixel is a photon traced backwards from the camera. A null geodesic in
Schwarzschild spacetime obeys

    d2x/dl2 = -(3/2) * h^2 * x / |x|^5        with  h = |x cross v|  (conserved)

integrated with RK4. Along the way the ray collects light from the optically
thin magnetosphere (dipole field lines, polar beams, an equatorial wind ring);
it ends either on the surface - thermal emission at ~10^6 K, gravitationally
redshifted, Doppler-boosted by the star's spin, limb-darkened - or in a
procedural star field that is lensed by the star's gravity.

A 10^6 K blackbody is deep in the Rayleigh-Jeans limit, so its colour is the
limit of the Planckian locus, CIE (x, y) = (0.2399, 0.2342): a blue-white.
That is the colour of the surface, not an artistic choice.

Outputs
    <out>.jpg          tone-mapped image used as the banner backdrop
    <out>.png          lossless copy for inspection
    <out>_flash.jpg    the star region at +0.9 stop, for the SVG pulse
    <out>_flash.json   where that crop sits, in viewBox units

Usage
    python scripts/render_neutronstar.py --scale 1.5 --ss 3 --out build/neutronstar
"""
import argparse
import json
import math
import os

import numpy as np
from numba import njit, prange
from PIL import Image
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------- scene ----
VIEW_W, VIEW_H = 1600, 640        # SVG viewBox the image is composed for
STAR_X, STAR_Y = 1196.0, 318.0    # where the star's centre sits in the viewBox
PX_PER_RS = 25.0                  # viewBox pixels per Schwarzschild radius
CAM_DIST = 60.0                   # camera distance, in r_s
R_NS = 2.5                        # stellar radius, in r_s
R_VOL = 30.0                      # magnetosphere is sampled inside this radius
R_ESC = 90.0                      # rays past this radius have escaped
BETA_EQ = 0.11                    # equatorial spin speed / c (~ 700 Hz)
APPARENT_R = R_NS / math.sqrt(1.0 - 1.0 / R_NS)   # lensed radius of the disc
FLASH_HALF = 250.0                # half-size of the pulse crop, viewBox px


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _axis(screen_tilt_deg, toward_cam_deg):
    """Unit vector: tilted right of screen-up, and toward the camera (-y)."""
    a, b = math.radians(screen_tilt_deg), math.radians(toward_cam_deg)
    return _unit([math.sin(a) * math.cos(b), -math.sin(b), math.cos(a) * math.cos(b)])


# camera sits on -y looking at the origin: screen x = world x, screen up = world z
MAG = _axis(31.0, 24.0)           # magnetic dipole axis (beams run along it)
SPIN = _axis(9.0, -27.0)          # rotation axis, top tipped away from us
_e1 = _unit(np.cross(MAG, [0.31, 0.77, 0.55]))
_e2 = np.cross(MAG, _e1)
_s1 = _unit(np.cross(SPIN, [0.0, 1.0, 0.0]))
_s2 = np.cross(SPIN, _s1)
MX, MY, MZ = (float(c) for c in MAG)
E1X, E1Y, E1Z = (float(c) for c in _e1)
E2X, E2Y, E2Z = (float(c) for c in _e2)
SX, SY, SZ = (float(c) for c in SPIN)
S1X, S1Y, S1Z = (float(c) for c in _s1)
S2X, S2Y, S2Z = (float(c) for c in _s2)

# Rayleigh-Jeans limit of the Planckian locus -> linear sRGB, blue = 1
_xc, _yc = 0.2399, 0.2342
_XYZ = np.array([_xc / _yc, 1.0, (1 - _xc - _yc) / _yc])
_M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
_rgb = _M @ _XYZ
HOT_R, HOT_G, HOT_B = (float(c) for c in _rgb / _rgb[2])

SURF_I = 23.0                     # surface brightness (HDR units)
CAP_CORE, CAP_HALO = 2.2, 0.5     # extra brightness of the magnetic polar caps
FIL_I = 0.030                     # field-line filaments
COR_I = 0.022                     # diffuse closed-zone plasma
BEAM_I = 1.10                     # polar beams
RING_I = 0.0                      # equatorial wind ring
SHELLS = (3.4, 4.6, 6.4, 8.8)                  # dipole shells L = r / sin^2(theta)
SHELL_W = (1.0, 0.8, 0.6, 0.42)
LINES_PER_SHELL = 5
THETA_PC = math.radians(6.5)      # half-opening of the open-field bundle at the surface
THETA_MAX = math.radians(8.5)     # beam stops flaring here
RING_R = 12.5

rng = np.random.default_rng(7)
PERM = np.concatenate([rng.permutation(256)] * 2).astype(np.int64)


# ---------------------------------------------------------------- noise ----
@njit(cache=True, fastmath=True)
def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True)
def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


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
def remnant(dx, dy, dz, perm):
    """The supernova remnant the pulsar was born in: wispy shock filaments.

    This layer is composited *unlensed*, along the camera's original ray. From
    180 km away the star's Einstein ring is ~11 r_s wide, and a lensed nebula
    wraps into a whirlpool that reads as a black hole - an honest render, but
    the wrong picture. The stars behind it stay fully lensed.

    Ridged, domain-warped noise gives thin sheets seen edge-on as filaments;
    colour drifts between doubly-ionised oxygen (teal) and hydrogen + blue
    continuum (violet), with dark dust lanes in front.
    """
    cx, cy, cz = 0.10, 0.99, 0.06                 # roughly behind the star
    cosang = dx * cx + dy * cy + dz * cz
    ang2 = 2.0 * (1.0 - cosang)
    env = math.exp(-ang2 / (2.0 * 0.36 * 0.36))
    if env < 0.01:
        return 0.0, 0.0, 0.0, 1.0
    w1 = fbm3(dx * 3.0 + 1.0, dy * 3.0, dz * 3.0, perm, 3)
    w2 = fbm3(dx * 3.0 + 7.0, dy * 3.0 + 2.0, dz * 3.0, perm, 3)
    ux = dx * 5.5 + 2.2 * w1
    uy = dy * 5.5 + 2.2 * w2
    uz = dz * 5.5 + 2.2 * w1
    r1 = 1.0 - abs(2.0 * fbm3(ux, uy, uz, perm, 5) - 1.0)
    r2 = 1.0 - abs(2.0 * fbm3(ux * 2.7 + 4.0, uy * 2.7, uz * 2.7, perm, 4) - 1.0)
    fil = r1 ** 6 * (0.5 + 0.9 * r2 ** 3) + 0.25 * r1 ** 2 * 0.15
    mix = smoothstep(0.35, 0.65, fbm3(dx * 2.0 + 9.0, dy * 2.0, dz * 2.0 + 4.0, perm, 3))
    vio = smoothstep(0.62, 0.8, fbm3(dx * 3.5 - 6.0, dy * 3.5, dz * 3.5, perm, 3))
    cr = 0.06 + 0.10 * mix + 0.22 * vio
    cg = 0.40 - 0.22 * mix - 0.12 * vio
    cb = 0.72 + 0.20 * mix
    k = 0.034 * env * fil
    dust = smoothstep(0.50, 0.72, fbm3(dx * 6.0 + 20.0, dy * 6.0, dz * 6.0, perm, 4))
    ext = 1.0 - 0.75 * dust * min(env * 1.6, 1.0)
    return k * cr * ext, k * cg * ext, k * cb * ext, ext


@njit(cache=True, fastmath=True)
def sky(dx, dy, dz, perm, star_sigma):
    # faint galactic band with dust lanes, cool blue
    bx, by, bz = 0.42, 0.38, -0.82            # band normal
    lat = dx * bx + dy * by + dz * bz + 0.08
    band = math.exp(-(lat * lat) / 0.02)
    n1 = fbm3(dx * 4.0 + 11.0, dy * 4.0, dz * 4.0, perm, 5)
    n2 = fbm3(dx * 10.0, dy * 10.0 + 5.0, dz * 10.0, perm, 4)
    lanes = min(max((n1 - 0.36) / 0.30, 0.0), 1.0)
    glow = band * _smooth(lanes) * (0.35 + 1.3 * n2)
    r = 0.004 * glow
    g = 0.007 * glow
    b = 0.018 * glow
    # a far-off violet nebula wash, very faint
    neb = fbm3(dx * 2.2 + 40.0, dy * 2.2 + 3.0, dz * 2.2, perm, 4)
    neb = max(0.0, neb - 0.52) * 1.9
    r += 0.005 * neb
    g += 0.003 * neb
    b += 0.010 * neb

    # hashed stars on a thin spherical shell of lattice cells
    S = 85.0
    px, py, pz = dx * S, dy * S, dz * S
    cx, cy, cz = int(math.floor(px)), int(math.floor(py)), int(math.floor(pz))
    density = 0.16 + 0.40 * band
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
                mag = 0.0045 * (u + 0.006) ** -1.25     # power-law brightness
                h = lcg(h)
                tcol = 4200.0 + 14000.0 * (((h >> 8) & 0xFFFF) / 65535.0) ** 0.8
                sr, sg, sb = blackbody(tcol)
                e = mag * math.exp(-w)
                r += e * sr
                g += e * sg
                b += e * sb
    return r, g, b


# -------------------------------------------------------------- surface ----
@njit(cache=True, fastmath=True)
def surface(px, py, pz, ox, oy, oz, perm):
    """Observed light from the surface point p, photon leaving along unit o."""
    r = math.sqrt(px * px + py * py + pz * pz)
    nx, ny, nz = px / r, py / r, pz / r
    mu = max(nx * ox + ny * oy + nz * oz, 0.0)

    # magnetic polar caps: heated by particles running down the open field lines
    c = abs(nx * MX + ny * MY + nz * MZ)
    ang = math.acos(min(c, 1.0))
    core = math.exp(-(ang / 0.15) ** 2)
    halo = math.exp(-(ang / 0.40) ** 2)
    rim = math.exp(-((ang - 0.27) / 0.035) ** 2)

    # crust: magnetically confined hot patches, domain-warped
    w = fbm3(nx * 2.0 + 5.0, ny * 2.0, nz * 2.0, perm, 3)
    n2 = fbm3(nx * 4.6 + 2.2 * w, ny * 4.6 - 1.4 * w, nz * 4.6 + 9.0, perm, 5)
    fine = fbm3(nx * 16.0, ny * 16.0 + 3.0, nz * 16.0, perm, 3)
    gran = fbm3(nx * 34.0 + 1.0, ny * 34.0, nz * 34.0 - 2.0, perm, 3)
    tex = (0.88 + 0.36 * (n2 - 0.5)) * (0.86 + 0.28 * fine) * (0.86 + 0.28 * gran)
    tex = max(tex, 0.35)

    inten = SURF_I * tex * (1.0 + CAP_CORE * core + CAP_HALO * halo + 0.35 * rim)

    # limb darkening of a magnetised hydrogen atmosphere
    inten *= 0.02 + 0.98 * mu ** 2.6

    # spin: Doppler factor in the local static frame, normalised to the
    # mean so only the asymmetry (approaching limb brighter) shows
    vx = BETA_EQ * (SY * nz - SZ * ny)
    vy = BETA_EQ * (SZ * nx - SX * nz)
    vz = BETA_EQ * (SX * ny - SY * nx)
    b2 = vx * vx + vy * vy + vz * vz
    gam = 1.0 / math.sqrt(1.0 - b2)
    dop = 1.0 / (gam * (1.0 - (vx * ox + vy * oy + vz * oz)))
    inten *= dop ** 3
    return inten * HOT_R, inten * HOT_G, inten * HOT_B


# --------------------------------------------------------- magnetosphere ----
@njit(cache=True, fastmath=True)
def volume(px, py, pz, perm):
    """Emissivity (linear RGB per unit length) of the magnetosphere at p."""
    r = math.sqrt(px * px + py * py + pz * pz)
    if r < R_NS:
        return 0.0, 0.0, 0.0
    c = (px * MX + py * MY + pz * MZ) / r
    s2 = max(1.0 - c * c, 1e-7)
    L = r / s2
    er = eg = eb = 0.0

    # thin hot plasma hugging the surface
    skin = math.exp(-(r - R_NS) / 0.30)
    er += 0.30 * skin * 0.40
    eg += 0.30 * skin * 0.58
    eb += 0.30 * skin

    # hot corona: pair plasma glowing violet-blue, thinning outwards
    cor2 = 0.11 * math.exp(-(r - R_NS) / 1.5) * (R_NS / r) ** 2
    er += cor2 * 0.46
    eg += cor2 * 0.40
    eb += cor2

    # diffuse plasma trapped on closed field lines
    if L < 11.0:
        cor = COR_I * (R_NS / r) ** 3 * (1.0 - smoothstep(7.5, 11.0, L))
        er += cor * 0.36
        eg += cor * 0.52
        eb += cor

    # plasma sheets on nested dipole shells (L = r / sin^2 theta); seen in
    # projection a thin shell is brightest along its silhouette, which draws
    # the familiar nested lobes. Fine fibres run along the field.
    phi = math.atan2(px * E2X + py * E2Y + pz * E2Z, px * E1X + py * E1Y + pz * E1Z)
    sint = math.sqrt(s2)
    grad = s2 * sint / math.sqrt(1.0 + 3.0 * c * c)     # |dL| -> distance
    sig = 0.020 + 0.006 * r
    for k in range(len(SHELLS)):
        if abs(L - SHELLS[k]) > 0.4 * SHELLS[k]:      # linearisation fails near the axis
            continue
        d1 = (L - SHELLS[k]) * grad
        if abs(d1) > 10.0 * sig:
            continue
        q = d1 * d1 / (2.0 * sig * sig)
        f = math.exp(-q) + 0.10 * math.exp(-q / 10.0)
        fib = fbm3(math.cos(phi) * 9.0 + 3.0 * k, math.sin(phi) * 9.0, 0.6 * k + (1.3 if c > 0.0 else 0.0), perm, 3)
        f *= FIL_I * SHELL_W[k] * (R_NS / r) ** 1.6 * (0.35 + 1.3 * fib * fib)
        er += f * 0.36
        eg += f * 0.55
        eb += f

    # polar beams: the open-field bundle, flaring like a dipole then collimating
    ac = abs(c)
    theta = math.acos(min(ac, 1.0))
    tb = math.asin(min(1.0, math.sin(THETA_PC) * math.sqrt(r / R_NS)))
    tb = min(tb, THETA_MAX)
    x = theta / tb
    if x < 1.8:
        prof = 2.2 * math.exp(-(x / 0.5) ** 2) + 5.0 * math.exp(-(x / 0.12) ** 2)    # sheath + hot spine
        rad = (1.0 - math.exp(-(r - R_NS) / 0.3)) * (R_NS / r) ** 1.7 * math.exp(-r / 22.0)
        hemi = 6.0 if c > 0.0 else 0.0
        lr = math.log(r)
        n = fbm3(math.cos(phi) * 7.0 + hemi, math.sin(phi) * 7.0, lr * 1.2 - hemi, perm, 4)
        knots = fbm3(math.cos(phi) * 1.4, math.sin(phi) * 1.4 + hemi, lr * 5.0, perm, 3)
        streak = (0.3 + 1.5 * n * n) * (0.75 + 0.5 * knots)
        j = BEAM_I * prof * rad * streak
        wht = math.exp(-(r - R_NS) / 2.0)
        er += j * (0.14 + 0.6 * wht)
        eg += j * (0.32 + 0.5 * wht)
        eb += j

    # equatorial ring of the pulsar wind, in the spin equator
    hh = px * SX + py * SY + pz * SZ
    rho = math.sqrt(max(r * r - hh * hh, 0.0))
    dr = rho - RING_R
    if RING_I > 0.0 and abs(dr) < 4.0 and abs(hh) < 2.0:
        psi = math.atan2(px * S2X + py * S2Y + pz * S2Z, px * S1X + py * S1Y + pz * S1Z)
        wisp = fbm3(math.cos(psi) * 5.0, math.sin(psi) * 5.0, dr * 0.7 + hh, perm, 4)
        ring = math.exp(-(dr * dr) / (2.0 * 0.85 * 0.85) - (hh * hh) / (2.0 * 0.28 * 0.28))
        ring *= RING_I * (0.15 + 2.2 * max(wisp - 0.32, 0.0))
        er += ring * 0.50
        eg += ring * 0.70
        eb += ring

    return er, eg, eb


# ------------------------------------------------------------ geodesics ----
@njit(cache=True, fastmath=True)
def trace(ox, oy, oz, dx, dy, dz, perm, star_sigma, jitter):
    x, y, z = ox, oy, oz
    vx, vy, vz = dx, dy, dz
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    k = -1.5 * (hx * hx + hy * hy + hz * hz)

    cr = cg = cb = 0.0
    entered = False
    for _ in range(8000):
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        if r > R_ESC and (x * vx + y * vy + z * vz) > 0.0:
            break
        if r < R_VOL:
            dt = max(0.0125 * r, 0.012)
            if not entered:                      # dither the step grid
                dt *= jitter
                entered = True
        else:
            dt = min(0.03 * r, 2.0)
            if r - dt < R_VOL and (x * vx + y * vy + z * vz) < 0.0:
                dt = max(r - R_VOL + 1e-3, 1e-3)

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
        rn = math.sqrt(nx * nx + ny * ny + nz * nz)

        if rn < R_NS:                            # hit the surface
            t = (r - R_NS) / (r - rn)
            sx = x + t * (nx - x)
            sy = y + t * (ny - y)
            sz = z + t * (nz - z)
            er, eg, eb = volume(0.5 * (x + sx), 0.5 * (y + sy), 0.5 * (z + sz), perm)
            cr += er * dt * t
            cg += eg * dt * t
            cb += eb * dt * t
            wx = vx + t * (nvx - vx)
            wy = vy + t * (nvy - vy)
            wz = vz + t * (nvz - vz)
            wn = math.sqrt(wx * wx + wy * wy + wz * wz)
            sr, sg, sb = surface(sx, sy, sz, -wx / wn, -wy / wn, -wz / wn, perm)
            return cr + sr, cg + sg, cb + sb

        if r < R_VOL:
            er, eg, eb = volume(0.5 * (x + nx), 0.5 * (y + ny), 0.5 * (z + nz), perm)
            cr += er * dt
            cg += eg * dt
            cb += eb * dt

        x, y, z = nx, ny, nz
        vx, vy, vz = nvx, nvy, nvz

    vn = math.sqrt(vx * vx + vy * vy + vz * vz)
    sr, sg, sb = sky(vx / vn, vy / vn, vz / vn, perm, star_sigma)
    nr, ng, nb, ext = remnant(dx, dy, dz, perm)
    return cr + sr * ext + nr, cg + sg * ext + ng, cb + sb * ext + nb


@njit(parallel=True, cache=True, fastmath=True)
def render(w, h, ss, cam, fwd, right, up, focal, cx, cy, perm, star_sigma):
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
                    hsh = hash3(i * 7 + si, j * 7 + sj, 91)
                    jit = 0.05 + 0.95 * (hsh & 0xFFFF) / 65535.0
                    r, g, b = trace(cam[0], cam[1], cam[2], dx / n, dy / n, dz / n,
                                    perm, star_sigma, jit)
                    ar += r
                    ag += g
                    ab += b
            img[j, i, 0] = ar * inv
            img[j, i, 1] = ag * inv
            img[j, i, 2] = ab * inv
    return img


# ------------------------------------------------------------- pipeline ----
def camera():
    cam = np.array([0.0, -CAM_DIST, 0.0])
    fwd = np.array([0.0, 1.0, 0.0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return cam, fwd, right, up


def glare(hdr, scale, strength=0.16):
    """Camera point-spread function: what makes a light source look *bright*.

    Real optics scatter a small fraction of the light into wide wings that
    fall off roughly as r^-3 (diffraction envelope + scatter). The wings are
    wider for longer wavelengths, so red spreads a little further than blue.
    Done as one FFT convolution per channel over a zero-padded frame.
    """
    h, w, _ = hdr.shape
    H, W = 2 * h, 2 * w
    yy = np.minimum(np.arange(H), H - np.arange(H))[:, None]
    xx = np.minimum(np.arange(W), W - np.arange(W))[None, :]
    r = np.hypot(yy, xx).astype(np.float64)
    out = np.empty_like(hdr)
    for c, lam in enumerate((1.25, 1.0, 0.80)):
        r0 = 1.6 * scale * lam
        k = (1.0 + (r / r0) ** 2) ** -1.45
        k[0, 0] = 0.0
        k /= k.sum()
        pad = np.zeros((H, W))
        pad[:h, :w] = hdr[..., c]
        conv = np.fft.irfft2(np.fft.rfft2(pad) * np.fft.rfft2(k), s=(H, W))
        out[..., c] = conv[:h, :w]
    return (1.0 - strength) * hdr + strength * out


# JWST's diffraction pattern: the hexagonal primary throws six spikes (three
# lines), and the vertical secondary-mirror strut adds a fainter horizontal
# pair. Angles in degrees from screen-right, with relative strength.
SPIKES = ((90.0, 1.0), (30.0, 1.0), (150.0, 1.0), (0.0, 0.32))


def spikes(hdr, scale, amp=2.4):
    """Razor-thin diffraction spikes, drawn as a point-source pattern.

    Convolving the whole resolved disc would smear them into bars, so they
    radiate from the star's centre. Spike length scales with wavelength,
    so the tips fade from blue-white into a faint red.
    """
    h, w, _ = hdr.shape
    cx, cy = STAR_X * scale, STAR_Y * scale
    disc = APPARENT_R * PX_PER_RS * scale
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    out = np.zeros_like(hdr, dtype=np.float64)
    width = 0.55 * scale
    for deg, k in SPIKES:
        a = math.radians(deg)
        along = dx * math.cos(a) - dy * math.sin(a)
        across = dx * math.sin(a) + dy * math.cos(a)
        line = np.exp(-0.5 * (across / width) ** 2)
        reach = np.maximum(np.abs(along) - disc * 0.8, 0.0)
        for c, lam in enumerate((1.2, 1.0, 0.85)):
            t0 = 10.0 * scale * lam
            prof = (1.0 + reach / t0) ** -1.7 * np.exp(-reach / (210.0 * scale * lam))
            out[..., c] += k * prof * line
    rr = np.hypot(dx, dy) / disc                      # lost in the saturated disc
    out *= np.clip((rr - 0.85) / 0.3, 0.0, 1.0)[..., None]
    tint = np.array([0.62, 0.76, 1.0])
    return hdr + amp * out * tint


# Lens ghosts: internal reflections imaged along the line from the source
# through the optical centre. g = 0 is the frame centre, g = -1 the source.
GHOSTS = ((-0.55, 20.0, (0.55, 0.85, 1.0), 0.011),
          (0.18, 9.0, (0.9, 0.6, 1.0), 0.013),
          (0.52, 34.0, (0.35, 0.75, 0.85), 0.006),
          (0.95, 15.0, (1.0, 0.72, 0.5), 0.007))


def ghosts(hdr, scale):
    """Faint hexagonal aperture ghosts with chromatic rims."""
    h, w, _ = hdr.shape
    cx, cy = w / 2.0, h / 2.0
    sx, sy = STAR_X * scale, STAR_Y * scale
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.zeros_like(hdr, dtype=np.float64)
    r3 = math.sqrt(3.0) / 2.0
    for g, size, tint, k in GHOSTS:
        px, py = cx + g * (cx - sx), cy + g * (cy - sy)
        x, y = xx - px, yy - py
        q = np.maximum(np.abs(y), np.maximum(np.abs(0.5 * y + r3 * x), np.abs(0.5 * y - r3 * x)))
        for c, lam in enumerate((1.04, 1.0, 0.96)):        # dispersion: fringed rims
            R = size * scale * lam
            body = np.clip((R - q) / (0.18 * R), 0.0, 1.0)
            rim = np.exp(-((q - R * 0.93) / (0.06 * R)) ** 2)
            out[..., c] += k * tint[c] * (0.6 * body + 0.5 * rim)
    return hdr + gaussian_filter(out, (1.2 * scale, 1.2 * scale, 0))


def srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def tonemap(hdr, scale, exposure=1.0):
    """Glare, a hue-preserving filmic curve, sensor-like highlight roll-off."""
    x = hdr.astype(np.float64) * exposure
    x = spikes(x, scale)
    x = glare(x, scale)
    x = ghosts(x, scale)
    for sigma, gain in ((3.0, 0.035), (14.0, 0.03)):          # atmospheric halo
        x += gain * gaussian_filter(x, (sigma * scale, sigma * scale, 0))

    h, w, _ = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - STAR_X * scale) / w, (yy - STAR_Y * scale) / w)
    x *= (1.0 - 0.45 * np.clip(d - 0.2, 0, 1) ** 1.3)[..., None]

    # filmic curve on luminance only, so blues stay blue instead of going grey
    L = x @ np.array([0.2126, 0.7152, 0.0722])
    Ld = L * (1.0 + L / 36.0) / (1.0 + L)
    x *= (Ld / np.maximum(L, 1e-9))[..., None]
    # a channel that clips bleeds into the others: hot things burn to white
    m = x.max(axis=2, keepdims=True)
    t = np.where(m > 1.0, 1.0 - np.power(np.maximum(m, 1.0), -1.15), 0.0)
    x = np.where(m > 1.0, x / np.maximum(m, 1e-9) * (1.0 - t) + t, x)

    floor = 0.0012                                            # true-black sky
    x = np.maximum(x - floor, 0.0) / (1.0 - floor)
    x = srgb(np.clip(x, 0.0, 1.0))
    x = x + (x * x * (3 - 2 * x) - x) * 0.12                # gentle S-curve
    # film grain is added as an SVG filter, which keeps the JPEG small
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.5, help="output px per viewBox px")
    ap.add_argument("--ss", type=int, default=2, help="supersampling per axis")
    ap.add_argument("--out", default="build/neutronstar")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--from-hdr", help="skip tracing, re-tonemap a saved *_hdr.npy")
    args = ap.parse_args()

    s = args.scale
    w, h = round(VIEW_W * s), round(VIEW_H * s)
    cam, fwd, right, up = camera()
    focal = PX_PER_RS * CAM_DIST * s
    star_sigma = 0.62 / focal

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.from_hdr:
        hdr = np.load(args.from_hdr)
    else:
        hdr = render(w, h, args.ss, cam, fwd, right, up, focal, STAR_X * s, STAR_Y * s, PERM, star_sigma)
        np.save(args.out + "_hdr.npy", hdr)
    base = tonemap(hdr, s, args.exposure)
    img = Image.fromarray(base)
    img.save(args.out + ".png")
    img.save(args.out + ".jpg", quality=args.quality, subsampling=0, optimize=True, progressive=True)

    # The pulse: the same frame exposed +0.9 stop, feathered back into the
    # base so the crossfade in the SVG is seamless at the crop edges.
    hot = tonemap(hdr, s, args.exposure * 1.9).astype(np.float64)
    half = FLASH_HALF
    x0, x1 = max(0, round((STAR_X - half) * s)), min(w, round((STAR_X + half) * s))
    y0, y1 = max(0, round((STAR_Y - half) * s)), min(h, round((STAR_Y + half) * s))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - STAR_X * s, yy - STAR_Y * s) / (half * s)
    wgt = (1.0 - np.clip((d - 0.35) / 0.6, 0.0, 1.0)) ** 2
    crop = base[y0:y1, x0:x1] * (1 - wgt[..., None]) + hot[y0:y1, x0:x1] * wgt[..., None]
    Image.fromarray((crop + 0.5).astype(np.uint8)).save(
        args.out + "_flash.jpg", quality=84, subsampling=0, optimize=True, progressive=True)
    with open(args.out + "_flash.json", "w") as f:
        json.dump({"x": x0 / s, "y": y0 / s, "w": (x1 - x0) / s, "h": (y1 - y0) / s}, f)
    print(f"wrote {args.out}  {w}x{h}  ss={args.ss}")


if __name__ == "__main__":
    main()
