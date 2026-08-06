# Motion Builder

Workspace for actually building a motion — raw input files, intermediate
conversions, and the script(s) that produce it, whether that's running a
single clip through `motion/video_to_motion/` or Kimodo, or splicing two
existing clips together. One subfolder per build. Split into `single/` and
`combined/`, mirroring `motion/motion_library/`'s own split.

Once a build's result is good, copy it into `motion/motion_library/` (under
`single/` or `combined/` as appropriate) with a README — see
`motion/motion_library/HOW_IT_WORKS.md` for the full pipeline this fits into.

## Current builds

- `single/wave_standing/` — leftover intermediate (`unused_alternates/`) from
  building `motion/motion_library/single/wave_standing/`.
- `single/wave_v2/` — leftover intermediate (`intermediate/`) from building
  `motion/motion_library/single/wave_v2/`.
- `combined/walk_and_wave/` — combines a walk clip + a wave clip (legs/root
  from one, arms from the other). Two results promoted so far:
  `motion/motion_library/combined/walk_wave/` and `.../walk_wave_v2/`
  (different wave source clips).

Note: several older single motions already in `motion/motion_library/single/`
(`cartwheel`, `gymnasts`, `walk_circle`) were built before this folder
existed, ad hoc, with no leftovers worth keeping — not backfilled here.

## Structure of a build folder (see `combined/walk_and_wave/` as the template)

```
<category>/<build_name>/
├── raw_npz/       ← raw source clips before conversion (if any)
├── converted/     ← each source clip, converted to ref_dof_pos format
├── combined/      ← output(s) — promoted into motion_library/ once good
└── scripts/       ← the build/combine script(s), plus any viewer helpers
```
