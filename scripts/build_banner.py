#!/usr/bin/env python3
"""Compose banner.svg and the link buttons.

GitHub renders README images in a sandbox: no scripts, no web fonts, no
external requests. So everything here is self-contained:

  * the ray-traced backdrop is embedded as a data URI
  * all text is converted to vector outlines (HarfBuzz shaping + fontTools)
  * motion is pure SMIL: typed terminal lines, twinkling stars and hot spots
    that orbit the black hole along their true gravitationally-lensed paths
    (taken from the ray tracer's own disk-crossing maps)

Usage
    python scripts/render_blackhole.py --out build/blackhole
    python scripts/build_banner.py --render build/blackhole
"""
import argparse
import base64
import math
import os

import numpy as np
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

from render_blackhole import DOPPLER_POWER, HOLE_X, HOLE_Y, PX_PER_RS, VIEW_H, VIEW_W

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONTS = os.path.join(HERE, "fonts")

INK = "#F4F0E8"       # primary text
SOFT = "#B9B4AA"      # secondary text
DIM = "#7F7B74"       # captions
AMBER = "#F5A85C"     # accent, sampled from the disk
VOID = "#04050A"


# ----------------------------------------------------------------- type ----
class Face:
    def __init__(self, filename):
        path = os.path.join(FONTS, filename)
        self.tt = TTFont(path)
        self.glyphs = self.tt.getGlyphSet()
        self.upem = self.tt["head"].unitsPerEm
        with open(path, "rb") as f:
            self.hb = hb.Font(hb.Face(f.read()))

    def outline(self, text, size, x, y, tracking=0.0, anchor="start", features=None):
        """Return (path data, advance width) for text set at (x, y) baseline."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, features or {})
        s = size / self.upem
        gap = tracking * size
        n = len(buf.glyph_infos)
        width = sum(p.x_advance for p in buf.glyph_positions) * s + gap * (n - 1)
        pen_x = x - {"start": 0.0, "middle": width / 2, "end": width}[anchor]
        out = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            pen = SVGPathPen(self.glyphs, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
            t = TransformPen(pen, (s, 0, 0, -s, pen_x + pos.x_offset * s, y - pos.y_offset * s))
            self.glyphs[self.tt.getGlyphName(info.codepoint)].draw(t)
            out.append(pen.getCommands())
            pen_x += pos.x_advance * s + gap
        return "".join(out), width


def path(d, fill, **attrs):
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<path d="{d}" fill="{fill}"{extra}/>'


def num(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------ hot spots ----
def lensed_orbit(maps, scale, radius, samples=96):
    """Screen-space path of a point orbiting at `radius`, as the camera sees it.

    The ray tracer stored, for every pixel, the radius/azimuth at which that
    pixel's ray first meets the disk plane. The iso-contour r = radius of that
    map is therefore the lensed image of a circular orbit; re-sampling it
    uniformly in azimuth gives positions at equal time steps.
    """
    r, phi, g = maps[..., 0], maps[..., 1], maps[..., 2]
    valid = r > 0
    contours = find_contours(np.where(valid, r, 0), radius, mask=valid)
    c = max(contours, key=len)
    if np.hypot(*(c[0] - c[-1])) > 2:
        raise RuntimeError("orbit contour is not closed")
    c = c[:-1]
    coords = [c[:, 0], c[:, 1]]
    cs = map_coordinates(np.cos(phi), coords, order=1)
    sn = map_coordinates(np.sin(phi), coords, order=1)
    gg = map_coordinates(g, coords, order=1)
    ang = np.unwrap(np.arctan2(sn, cs))
    if ang[-1] < ang[0]:                       # make azimuth increase = prograde
        c, ang, gg = c[::-1], -ang[::-1], gg[::-1]
        ang = np.unwrap(-ang)
        if ang[-1] < ang[0]:
            ang = -ang
    start = int(np.argmin(ang % (2 * math.pi)))
    c, gg = np.roll(c, -start, axis=0), np.roll(gg, -start)
    ang = np.unwrap(np.roll(ang, -start))
    ang -= ang[0]
    c = np.vstack([c, c[:1]])
    gg = np.append(gg, gg[0])
    ang = np.append(ang, 2 * math.pi)
    t = np.linspace(0, 2 * math.pi, samples + 1)
    xs = np.interp(t, ang, c[:, 1]) / scale
    ys = np.interp(t, ang, c[:, 0]) / scale
    beam = np.interp(t, ang, gg) ** DOPPLER_POWER
    return xs, ys, beam / beam.max()


def hot_spot(maps, scale, radius, period, phase, size, strength):
    xs, ys, beam = lensed_orbit(maps, scale, radius)
    pos = ";".join(f"{num(x)},{num(y)}" for x, y in zip(xs, ys))
    out = []
    for i in range(8):                          # head + fading tail
        lag = phase + i * 0.0045 * period
        fade = strength * (1 - i / 8) ** 1.6
        op = ";".join(f"{max(0.02, b) * fade:.2f}" for b in beam)
        out.append(
            f'<g opacity="0"><animateTransform attributeName="transform" type="translate" '
            f'values="{pos}" dur="{period}s" begin="-{lag:.2f}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{op}" dur="{period}s" begin="-{lag:.2f}s" '
            f'repeatCount="indefinite"/><circle r="{num(size * (1 - i * 0.08))}" fill="url(#hot)"/></g>'
        )
    return "".join(out)


# --------------------------------------------------------------- typing ----
def typed_lines(face, lines, size, x, y, color):
    """Terminal lines that type in, hold, and backspace - one after another."""
    adv = face.tt["hmtx"][face.tt.getGlyphName(hb_gid(face, "m"))][0] * size / face.upem
    type_dt, hold, erase_dt, gap = 0.058, 2.9, 0.022, 0.55
    t, tracks = 0.0, []
    for line in lines:
        ev = []
        for k in range(1, len(line) + 1):
            ev.append((t, k))
            t += type_dt
        t += hold
        for k in range(len(line) - 1, -1, -1):
            ev.append((t, k))
            t += erase_dt
        t += gap
        tracks.append(ev)
    total = t

    def anim(attr, events, fmt):
        if events[0][0] > 0:
            events = [(0.0, 0)] + events
        vals = ";".join(fmt(k) for _, k in events)
        keys = ";".join(f"{tt / total:.5f}" for tt, _ in events)
        return (f'<animate attributeName="{attr}" calcMode="discrete" values="{vals}" '
                f'keyTimes="{keys}" dur="{total:.2f}s" repeatCount="indefinite"/>')

    defs, body = [], []
    for i, (line, ev) in enumerate(zip(lines, tracks)):
        d, _ = face.outline(line, size, x, y, features={"liga": False, "calt": False})
        defs.append(f'<clipPath id="type{i}"><rect x="{num(x - 2)}" y="{num(y - size)}" width="0" '
                    f'height="{num(size * 1.5)}">{anim("width", ev, lambda k: num(k * adv + 2))}</rect></clipPath>')
        body.append(f'<g clip-path="url(#type{i})">{path(d, color)}</g>')
    merged = [e for ev in tracks for e in ev]
    body.append(
        f'<rect x="{num(x)}" y="{num(y - size * 0.82)}" width="{num(adv * 0.56)}" height="{num(size * 1.02)}" '
        f'fill="{AMBER}">{anim("x", merged, lambda k: num(x + k * adv + 2))}'
        f'<animate attributeName="opacity" calcMode="discrete" values="1;0" keyTimes="0;0.55" dur="1.1s" '
        f'repeatCount="indefinite"/></rect>')
    return "".join(defs), "".join(body)


def hb_gid(face, ch):
    return face.hb.get_nominal_glyph(ord(ch))


# ---------------------------------------------------------------- stars ----
def twinkles(count=26):
    rng = np.random.default_rng(11)
    out = []
    while len(out) < count:
        x, y = rng.uniform(30, VIEW_W - 30), rng.uniform(26, VIEW_H - 26)
        in_text = 60 < x < 800 and 150 < y < 500
        in_disk = math.hypot((x - HOLE_X) / 470, (y - HOLE_Y) / 215) < 1
        if in_text or in_disk:
            continue
        r = rng.choice([0.9, 1.1, 1.4, 1.8], p=[0.4, 0.3, 0.2, 0.1])
        hi, dur, beg = rng.uniform(0.55, 0.95), rng.uniform(2.6, 6.5), rng.uniform(0, 6)
        tint = rng.choice(["#FFFFFF", "#DCE6FF", "#FFE9CF"])
        out.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{r}" fill="{tint}" opacity="{hi * 0.25:.2f}">'
                   f'<animate attributeName="opacity" values="{hi * 0.25:.2f};{hi:.2f};{hi * 0.25:.2f}" '
                   f'dur="{dur:.1f}s" begin="-{beg:.1f}s" repeatCount="indefinite"/></circle>')
    return "".join(out)


# --------------------------------------------------------------- banner ----
def build_banner(render, out):
    jost_l, jost_r = Face("Jost-Light.ttf"), Face("Jost-Regular.ttf")
    mono, mono_m = Face("JetBrainsMono-Regular.ttf"), Face("JetBrainsMono-Medium.ttf")
    npz = np.load(render + "_maps.npz")
    maps, mscale = npz["maps"], float(npz["scale"])
    with open(render + ".jpg", "rb") as f:
        photo = base64.b64encode(f.read()).decode()

    W, H, L = VIEW_W, VIEW_H, 92                         # L = left margin
    els, defs = [], []

    # -- hero text ---------------------------------------------------------
    d, _ = mono_m.outline("PROFESSIONAL STARTER OF PROJECTS", 19, L, 212, tracking=0.30)
    els.append(path(d, AMBER))
    d, name_w = jost_l.outline("KUSHAL YADAV", 94, L - 4, 318, tracking=0.092)
    els.append(path(d, INK))
    els.append(f'<rect x="{L}" y="356" width="64" height="2" fill="{AMBER}"/>')
    d, _ = jost_r.outline("Building things nobody asked for.", 37, L, 416, tracking=0.012)
    els.append(path(d, INK, opacity="0.94"))

    d, pw = mono_m.outline(">", 23, L, 474)
    els.append(path(d, AMBER))
    tdefs, tbody = typed_lines(mono, [
        "Thinks a lot. Pushes less.",
        "I turn bugs into features.",
        'git commit -m "final v2 REAL"',
    ], 23, L + pw + 13, 474, SOFT)
    defs.append(tdefs)
    els.append(tbody)

    # -- top + bottom rails --------------------------------------------------
    d, _ = mono.outline("GITHUB.COM/KUSHALXYADAV", 16.5, L, 70, tracking=0.2)
    els.append(path(d, DIM))
    d, sw = mono.outline("STATUS  /  PRODUCTIVELY PROCRASTINATING", 16.5, W - L, 70, tracking=0.2, anchor="end")
    els.append(path(d, SOFT))
    dot_x = W - L - sw - 20
    els.append(
        f'<circle cx="{num(dot_x)}" cy="64.5" r="4" fill="none" stroke="{AMBER}" stroke-width="1.2">'
        f'<animate attributeName="r" values="4;11" dur="2.4s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0.9;0" dur="2.4s" repeatCount="indefinite"/></circle>'
        f'<circle cx="{num(dot_x)}" cy="64.5" r="4" fill="{AMBER}"/>')
    d, _ = mono.outline("KUSHALYADAV.ME", 16.5, L, H - 56, tracking=0.2)
    els.append(path(d, DIM))
    d, _ = mono.outline("FIG. 01  /  SCHWARZSCHILD BLACK HOLE, RAY-TRACED  /  i = 81.5°",
                        16.5, W - L, H - 56, tracking=0.2, anchor="end")
    els.append(path(d, DIM))

    # -- annotation: leader line from the horizon ---------------------------
    shadow = 2.598 * PX_PER_RS                              # critical impact parameter
    a = math.radians(58)
    ax, ay = HOLE_X + shadow * 0.9 * math.cos(a), HOLE_Y + shadow * 0.9 * math.sin(a)
    kx, ky, lx = 1296, 487, 1318
    els.append(
        f'<g fill="none" stroke="{INK}" stroke-width="1.2" opacity="0.8">'
        f'<path d="M{num(ax)} {num(ay)}L{kx} {ky}H{lx - 8}"/>'
        f'<circle cx="{num(ax)}" cy="{num(ay)}" r="4" fill="{VOID}"/></g>')
    d, _ = mono_m.outline("EVENT HORIZON", 16.5, lx, ky + 5, tracking=0.24)
    els.append(path(d, INK))
    for i, line in enumerate(("No side project has", "ever returned.")):
        d, _ = jost_r.outline(line, 20, lx, ky + 33 + i * 25, tracking=0.01)
        els.append(path(d, SOFT))

    # -- hot spots on lensed orbits ------------------------------------------
    spots = "".join([
        hot_spot(maps, mscale, 4.3, 17, 0.0, 17, 0.9),
        hot_spot(maps, mscale, 6.4, 31, 9.0, 16, 0.85),
        hot_spot(maps, mscale, 9.2, 53, 30.0, 15, 0.8),
    ])

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Kushal Yadav</title>
<desc id="d">Kushal Yadav, professional starter of projects. Building things nobody asked for. A ray-traced black hole with a glowing accretion disk sits on the right.</desc>
<defs>
<clipPath id="frame"><rect width="{W}" height="{H}" rx="22"/></clipPath>
<linearGradient id="scrim" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{VOID}" stop-opacity=".55"/><stop offset=".36" stop-color="{VOID}" stop-opacity=".34"/><stop offset=".56" stop-color="{VOID}" stop-opacity="0"/></linearGradient>
<linearGradient id="rails" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{VOID}" stop-opacity=".55"/><stop offset=".2" stop-color="{VOID}" stop-opacity="0"/><stop offset=".8" stop-color="{VOID}" stop-opacity="0"/><stop offset="1" stop-color="{VOID}" stop-opacity=".6"/></linearGradient>
<radialGradient id="hot"><stop offset="0" stop-color="#FFF6E0" stop-opacity=".95"/><stop offset=".35" stop-color="#FFC982" stop-opacity=".5"/><stop offset="1" stop-color="#FF9A3D" stop-opacity="0"/></radialGradient>
{"".join(defs)}
</defs>
<g clip-path="url(#frame)">
<rect width="{W}" height="{H}" fill="{VOID}"/>
<image width="{W}" height="{H}" preserveAspectRatio="xMidYMid slice" href="data:image/jpeg;base64,{photo}"/>
<g>{twinkles()}</g>
<g style="mix-blend-mode:screen">{spots}</g>
<rect width="{W}" height="{H}" fill="url(#scrim)"/>
<rect width="{W}" height="{H}" fill="url(#rails)"/>
{"".join(els)}
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="21.5" fill="none" stroke="#FFFFFF" stroke-opacity=".09"/>
</g>
</svg>
'''
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}  {os.path.getsize(out) / 1024:.0f} KB")


# ---------------------------------------------------------------- links ----
def build_link(label, out):
    mono_m = Face("JetBrainsMono-Medium.ttf")
    size, pad, h = 13, 18, 38
    d, w = mono_m.outline(label, size, pad, h / 2 + size * 0.36, tracking=0.16)
    ax = pad + w + 12                                    # arrow origin
    total = ax + 9 + pad
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(total)} {h}" width="{num(total)}" height="{h}" '
           f'role="img" aria-label="{label.title()}"><title>{label.title()}</title>'
           f'<rect x=".5" y=".5" width="{num(total - 1)}" height="{h - 1}" rx="9.5" fill="#07080D" stroke="#34353D"/>'
           f'{path(d, INK)}'
           f'<path d="M{num(ax)} {h / 2 + 4.5}l9-9m-6.5 0h6.5v6.5" fill="none" stroke="{AMBER}" stroke-width="1.6" '
           f'stroke-linecap="round" stroke-linejoin="round"/></svg>\n')
    with open(out, "w") as f:
        f.write(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=os.path.join(ROOT, "build", "blackhole"))
    args = ap.parse_args()
    build_banner(args.render, os.path.join(ROOT, "banner.svg"))
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    for label in ("PORTFOLIO", "EMAIL", "LINKEDIN", "INSTAGRAM", "X"):
        build_link(label, os.path.join(ROOT, "assets", f"link-{label.lower()}.svg"))


if __name__ == "__main__":
    main()
