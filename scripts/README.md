# How the banner is made

`banner.svg` is generated, not drawn.

1. **`render_blackhole.py`** traces one photon per sub-pixel backwards from the
   camera through Schwarzschild spacetime (RK4 on the null-geodesic equation).
   Rays that fall past the horizon form the shadow; rays that cross the
   equatorial plane light up the accretion disk, with relativistic Doppler
   beaming and gravitational redshift; rays that escape sample a procedural
   star field, which is why the stars near the hole smear into arcs.
2. **`build_banner.py`** embeds that render in an SVG, converts every line of
   text to vector outlines (so it looks identical everywhere, no web fonts),
   and adds SMIL animation: the typed terminal lines, twinkling stars, and hot
   spots that orbit along their real gravitationally-lensed paths — over the
   top of the shadow when they are behind the hole.

```bash
pip install -r scripts/requirements.txt
```

```bash
python scripts/render_blackhole.py --scale 1.5 --ss 3 --out build/blackhole
```

```bash
python scripts/build_banner.py --render build/blackhole
```

Fonts in `scripts/fonts/` are subsets of [Jost](https://github.com/indestructible-type/Jost)
and [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono), both under the SIL Open Font License.
