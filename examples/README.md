# Example decks

Small, self-contained LS-DYNA keyword decks you can run the tool against.
Each parses, auto-detects to the unit system in its header, and reports as
`convertible` under `kunit check`.

| File | Units | What's in it |
|------|-------|--------------|
| `steel_plate_kg-m-s.k` | kg-m-s (SI) | Steel `*MAT_PIECEWISE_LINEAR_PLASTICITY` (MAT_024, ρ=7850, E=2.1e11), a shell part, four `*NODE`s, and a gravity-shaped `*DEFINE_CURVE` driven by `*LOAD_BODY_Z` (9.80665 m/s²). Exercises curve-dimension resolution. |
| `steel_coupon_ton-mm-s.k` | ton-mm-s | The same steel material expressed in ton-mm-s (ρ=7.85e-9, E=2.1e5 MPa, σ_y=250) — a minimal deck in the other common crash unit system. |
| `aluminium_bar_kg-m-s.k` | kg-m-s (SI) | Elastic aluminium (`*MAT_ELASTIC`, ρ=2700, E=7.0e10) solid part with four nodes — the simplest detectable deck. |

## Try them

```bash
kunit detect  examples/steel_plate_kg-m-s.k        # auto-detects kg-m-s
kunit check   examples/steel_plate_kg-m-s.k        # coverage / convertibility
kunit convert examples/steel_plate_kg-m-s.k --to ton-mm-s --dry-run

# without installing:
python kunit.py detect examples/aluminium_bar_kg-m-s.k
```
