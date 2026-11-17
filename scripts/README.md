# How the banner is made

`banner.svg` is generated, not drawn.

1. **`render_neutronstar.py`** traces one photon per sub-pixel backwards from
   the camera through Schwarzschild spacetime (RK4 on the null-geodesic
   equation) around a neutron star of radius 2.5 r_s — about 10 km for
   1.4 solar masses.
   - Gravity bends light so hard that the camera sees ~80 % of the surface at
     once, both magnetic poles included.
   - The surface is a 10^6 K blackbody. That deep in the Rayleigh–Jeans limit
     its colour is the end of the Planckian locus, CIE (x, y) = (0.2399, 0.2342):
     the blue-white is physics, not a colour pick. It is limb-darkened,
     Doppler-boosted by the star's spin, with hot magnetic polar caps.
   - On the way in, each ray collects light from the optically thin
     magnetosphere: thin plasma sheets on nested dipole shells (L = r / sin²θ), the
     open-field bundle that becomes the two polar beams.
   - Rays that miss the star sample a procedural star field, lensed by the
     star's gravity — which is why stars nearby smear into arcs. Behind it
     sits the supernova remnant it was born in (wispy, ridged-noise
     filaments). That layer is composited unlensed: from this close the
     Einstein ring is ~11 r_s wide and would wrap the nebula into a whirlpool.
   - The camera is simulated too: a point-spread function with r^-3 wings
     (wider in red than blue), JWST's diffraction pattern (six spikes from a
     hexagonal primary, a faint horizontal pair from the secondary's strut),
     faint hexagonal lens ghosts along the optical axis, and a filmic curve on
     luminance where a clipping channel bleeds to white — so the star burns
     white-hot in the middle and stays blue at the limb, like a real sensor.
2. **`build_banner.py`** embeds that render in an SVG, converts every line of
   text to vector outlines (so it looks identical everywhere, no web fonts),
   and adds SMIL animation: the typed terminal lines, twinkling stars, and the
   pulsar's double flash — main pulse plus interpulse, one from each pole.
   The flash is no drawn glow: it crossfades to the same render exposed
   +0.9 stop. Film grain is an `feTurbulence` filter rather than pixels in the
   JPEG, which keeps the whole banner under 250 KB.

```bash
pip install -r scripts/requirements.txt
```

```bash
python scripts/render_neutronstar.py --scale 1.5 --ss 3 --out build/neutronstar
```

```bash
python scripts/build_banner.py --render build/neutronstar
```

Fonts in `scripts/fonts/` are subsets of [Jost](https://github.com/indestructible-type/Jost)
and [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono), both under the SIL Open Font License.
