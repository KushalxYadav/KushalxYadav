#!/usr/bin/env python3
"""Compose banner.svg and the link buttons.

GitHub renders README images in a sandbox: no scripts, no web fonts, no
external requests. So everything here is self-contained:

  * the ray-traced backdrop is embedded as a data URI
  * all text is converted to vector outlines (HarfBuzz shaping + fontTools)
  * motion is pure SMIL: typed terminal lines, twinkling stars, and the
    pulsar itself - a double flash per turn (main pulse + interpulse, one
    from each magnetic pole), crossfading to the render at +0.9 stop
  * film grain is an feTurbulence filter, not baked into the JPEG

Usage
    python scripts/render_neutronstar.py --out build/neutronstar
    python scripts/build_banner.py --render build/neutronstar
"""
import argparse
import base64
import json
import math
import os

import numpy as np
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

from render_neutronstar import APPARENT_R, PX_PER_RS, STAR_X, STAR_Y, VIEW_H, VIEW_W

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONTS = os.path.join(HERE, "fonts")

INK = "#EEF3FB"       # primary text
SOFT = "#B3BCCB"      # secondary text
DIM = "#79818F"       # captions
ICE = "#9CC3FF"       # accent, sampled from the star's limb
VOID = "#000000"
DISC = APPARENT_R * PX_PER_RS   # radius of the star's lensed disc, viewBox px


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


# --------------------------------------------------------------- pulsar ----
PERIOD = 1.6     # seconds per turn on screen (the real thing: ~1.4 ms)


def pulsar(render):
    """Lighthouse flashes: main pulse, then the weaker interpulse half a turn later.

    No drawn glow: the flash is the ray-traced frame itself at +0.9 stop,
    cropped around the star and crossfaded in.
    """
    with open(render + "_flash.json") as f:
        box = json.load(f)
    with open(render + "_flash.jpg", "rb") as f:
        hot = base64.b64encode(f.read()).decode()
    keys = "0;0.06;0.18;0.5;0.56;0.66;1"
    return (f'<image x="{num(box["x"])}" y="{num(box["y"])}" width="{num(box["w"])}" height="{num(box["h"])}" '
            f'preserveAspectRatio="none" opacity="0" href="data:image/jpeg;base64,{hot}">'
            f'<animate attributeName="opacity" values="0;1;0;0;.5;0;0" keyTimes="{keys}" '
            f'calcMode="spline" keySplines="0 0 .2 1;.4 0 .6 1;0 0 1 1;0 0 .2 1;.4 0 .6 1;0 0 1 1" '
            f'dur="{PERIOD}s" repeatCount="indefinite"/></image>')


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
        f'fill="{ICE}">{anim("x", merged, lambda k: num(x + k * adv + 2))}'
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
        in_text = 60 < x < 820 and 150 < y < 500
        near_star = math.hypot(x - STAR_X, y - STAR_Y) < DISC * 3.2
        if in_text or near_star:
            continue
        r = rng.choice([0.9, 1.1, 1.4, 1.8], p=[0.4, 0.3, 0.2, 0.1])
        hi, dur, beg = rng.uniform(0.55, 0.95), rng.uniform(2.6, 6.5), rng.uniform(0, 6)
        tint = rng.choice(["#FFFFFF", "#DCE6FF", "#C9DAFF"])
        out.append(f'<circle cx="{num(x)}" cy="{num(y)}" r="{r}" fill="{tint}" opacity="{hi * 0.25:.2f}">'
                   f'<animate attributeName="opacity" values="{hi * 0.25:.2f};{hi:.2f};{hi * 0.25:.2f}" '
                   f'dur="{dur:.1f}s" begin="-{beg:.1f}s" repeatCount="indefinite"/></circle>')
    return "".join(out)


# --------------------------------------------------------------- banner ----
def build_banner(render, out):
    jost_l, jost_r = Face("Jost-Light.ttf"), Face("Jost-Regular.ttf")
    mono, mono_m = Face("JetBrainsMono-Regular.ttf"), Face("JetBrainsMono-Medium.ttf")
    with open(render + ".jpg", "rb") as f:
        photo = base64.b64encode(f.read()).decode()

    W, H, L = VIEW_W, VIEW_H, 92                         # L = left margin
    els, defs = [], []

    # -- hero text ---------------------------------------------------------
    d, _ = mono_m.outline("PROFESSIONAL STARTER OF PROJECTS", 19, L, 212, tracking=0.30)
    els.append(path(d, ICE))
    d, name_w = jost_l.outline("KUSHAL YADAV", 94, L - 4, 318, tracking=0.092)
    els.append(path(d, INK))
    els.append(f'<rect x="{L}" y="356" width="64" height="2" fill="{ICE}"/>')
    d, _ = jost_r.outline("Building things nobody asked for.", 37, L, 416, tracking=0.012)
    els.append(path(d, INK, opacity="0.94"))

    d, pw = mono_m.outline(">", 23, L, 474)
    els.append(path(d, ICE))
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
        f'<circle cx="{num(dot_x)}" cy="64.5" r="4" fill="none" stroke="{ICE}" stroke-width="1.2">'
        f'<animate attributeName="r" values="4;11" dur="{PERIOD * 1.5}s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0.9;0" dur="{PERIOD * 1.5}s" repeatCount="indefinite"/></circle>'
        f'<circle cx="{num(dot_x)}" cy="64.5" r="4" fill="{ICE}"/>')
    d, _ = mono.outline("KUSHALYADAV.ME", 16.5, L, H - 56, tracking=0.2)
    els.append(path(d, DIM))
    d, _ = mono.outline("FIG. 02  /  PULSAR, RAY-TRACED  /  R = 10 KM",
                        16.5, W - L, H - 56, tracking=0.2, anchor="end")
    els.append(path(d, DIM))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Kushal Yadav</title>
<desc id="d">Kushal Yadav, professional starter of projects. Building things nobody asked for. A ray-traced pulsar - a white-hot neutron star with a violet corona, twin beams and six-point diffraction spikes - sits on the right.</desc>
<defs>
<clipPath id="frame"><rect width="{W}" height="{H}" rx="22"/></clipPath>
<linearGradient id="scrim" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{VOID}" stop-opacity=".55"/><stop offset=".36" stop-color="{VOID}" stop-opacity=".34"/><stop offset=".56" stop-color="{VOID}" stop-opacity="0"/></linearGradient>
<linearGradient id="rails" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{VOID}" stop-opacity=".6"/><stop offset=".2" stop-color="{VOID}" stop-opacity="0"/><stop offset=".8" stop-color="{VOID}" stop-opacity="0"/><stop offset="1" stop-color="{VOID}" stop-opacity=".6"/></linearGradient>
<filter id="grain" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB"><feTurbulence type="fractalNoise" baseFrequency=".92" numOctaves="2" seed="7" stitchTiles="stitch"/><feColorMatrix values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1.9 -.78"/></filter>
<filter id="grain2" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB"><feTurbulence type="fractalNoise" baseFrequency=".92" numOctaves="2" seed="29" stitchTiles="stitch"/><feColorMatrix values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1.9 -.78"/></filter>
{"".join(defs)}
</defs>
<g clip-path="url(#frame)">
<rect width="{W}" height="{H}" fill="{VOID}"/>
<image width="{W}" height="{H}" preserveAspectRatio="xMidYMid slice" href="data:image/jpeg;base64,{photo}"/>
<g>{twinkles()}</g>
{pulsar(render)}
<rect width="{W}" height="{H}" fill="url(#scrim)"/>
<rect width="{W}" height="{H}" fill="url(#rails)"/>
<rect width="{W}" height="{H}" filter="url(#grain)" opacity=".09"/>
<rect width="{W}" height="{H}" filter="url(#grain2)" opacity=".14"/>
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
           f'<path d="M{num(ax)} {h / 2 + 4.5}l9-9m-6.5 0h6.5v6.5" fill="none" stroke="{ICE}" stroke-width="1.6" '
           f'stroke-linecap="round" stroke-linejoin="round"/></svg>\n')
    with open(out, "w") as f:
        f.write(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=os.path.join(ROOT, "build", "neutronstar"))
    args = ap.parse_args()
    build_banner(args.render, os.path.join(ROOT, "banner.svg"))
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    for label in ("PORTFOLIO", "EMAIL", "LINKEDIN", "INSTAGRAM", "X"):
        build_link(label, os.path.join(ROOT, "assets", f"link-{label.lower()}.svg"))


if __name__ == "__main__":
    main()
