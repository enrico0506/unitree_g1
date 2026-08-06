# Motion Library

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the full pipeline: what tools exist,
which source paths (video / Kimodo) actually work end-to-end today, and how a
clip goes from raw input to a trainable reference motion.

Reusable HoloMotion reference clips (`ref_dof_pos` [T,29] + `ref_global_*` [T,30,...] format,
G1 29-DOF). Two categories:

- `single/` — one atomic motion each (an ingredient). Nothing in here depends on
  anything else in the library.
- `combined/` — clips built by splicing/blending two or more `single/` clips
  together (a product). Each combined README cites exactly which `single/`
  clips and which script built it. Never build a combined clip from another
  combined clip — keeps the dependency graph one level deep.

Detail (fps, frame count, source, how it was made, known quirks) lives in each
motion's own `README.md`, right next to its data. This file is just an index.

## single/

| motion | fps | frames | duration | one-liner |
|---|---|---|---|---|
| [wave_standing](single/wave_standing/README.md) | 30 | 155 | 5.13s | standing, one-arm wave, gesture A (video-captured) |
| [wave_v2](single/wave_v2/README.md) | 30 | 198 | 6.6s | standing, one-arm wave, gesture B — distinct from wave_standing, not a replacement (video-captured) |
| [walk_circle](single/walk_circle/README.md) | 30 | 180 | 6.0s | walking in a curved path (Kimodo-generated) |
| [cartwheel](single/cartwheel/README.md) | 30 | 151 | 5.0s | cartwheel (TWIST example library) |
| [gymnasts](single/gymnasts/README.md) | 30 | 99 | 3.27s | gymnastics move (TWIST example library) |

## combined/

| motion | built from | one-liner |
|---|---|---|
| [walk_wave](combined/walk_wave/README.md) | walk_circle (legs/root) + wave_standing (arms) | walks while raising/waving/lowering arm |
| [walk_wave_v2](combined/walk_wave_v2/README.md) | walk_circle (legs/root) + wave_v2 (arms, resampled) | walks while raising/waving/lowering arm, different gesture from walk_wave |
