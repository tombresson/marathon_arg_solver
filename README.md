# Match Tile Assistant

`matchtile` is a Windows-first assistant for the Discord Match Tile activity.
It now uses explicit manual calibration instead of automatic board detection:

- open or create a saved calibration profile for the current Discord window size
- inspect the projected grid and adjust rows, cols, and group size in place
- re-pick corners from the same calibration window when the overlay looks wrong
- capture the player-specific reveal burst from Discord
- reconstruct tile identities from local frames
- show a transparent overlay with group IDs and confidence

## Environment

The project targets Python 3.13 because the desktop and CV dependencies are more predictable there than on 3.14.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Commands

```powershell
matchtile calibrate --window "NULL//TRANSMIT.ERR" --rows 25 --cols 64 --group-size 4
matchtile calibrate --image empty-board-2.png --rows 25 --cols 64 --group-size 4
matchtile replay --session sessions\20260319-024228
matchtile replay --board-json tmp_live_board.json --calibration calibrations\NULL_TRANSMIT.ERR_2556x1360.calibration.json
matchtile replay --images "Screenshot 2026-03-19 005448.png" --calibration empty-board-2.calibration.json
matchtile arm --window "NULL//TRANSMIT.ERR" --overlay
matchtile overlay --session sessions\20260319-013000 --window "NULL//TRANSMIT.ERR"
```

## Notes

- `calibrate` is the editor for saved calibration profiles. If a profile already exists for the current Discord client size, it opens immediately and shows the projected grid overlay; otherwise it starts in fresh corner-pick mode.
- Fresh calibration needs `--rows` and `--cols`. Existing profiles can be reopened without re-entering counts.
- In the calibration editor, use `Left`/`Right` for columns, `Up`/`Down` for rows, `-`/`=` for max group size, `C` to re-pick corners, `Backspace` to undo in corner mode, `R` to reset, `Enter`/`Space` to save, and `Esc` to cancel.
- Corner-pick mode now includes a live magnifier inset with crosshair and pixel coordinates to make precise corner placement easier.
- Live window calibrations are saved under `calibrations/` and reused automatically by `arm` for the same Discord client size.
- `arm` no longer auto-calibrates. If no matching calibration exists, it stops and tells you to run `matchtile calibrate` first.
- `arm` waits for `F8`, captures a fixed reveal burst from Discord, reconstructs the board locally, and uses the saved calibration profile's `rows`, `cols`, and max group size. A max group size of `3` allows both 2- and 3-matches; `4` allows 2-, 3-, and 4-matches.
- Runtime settings come from [matchtile.json](e:/Sync/Projects/Marathon%20ARG/matchtile.json) when that file exists. Editing [config.py](e:/Sync/Projects/Marathon%20ARG/matchtile/config.py) only changes defaults for new or missing config values.
- `arm` prints the active config file plus the effective `capture_fps` and `reveal_duration_s` at startup so you can confirm the live capture timing being used.
- `replay --board-json` is still available for offline website-board debugging. `replay --session` uses `session\calibration.json` when present.
- Use `--max-group-size <2-4>` on `replay`, `arm`, or `overlay` to cap inferred match-group size for a specific run.
- `F12` is a global emergency stop for `arm`, overlay mode, and auto-click mode.
- `arm --autoplay` clicks only high-confidence match groups by default and aborts if the post-click verification does not detect the expected board change.
- Autoplay now uses a move-settle and mouse-hold click primitive by default: `click_delay_s = 0.25`, `move_settle_s = 0.035`, `mouse_down_hold_s = 0.045`, `timing_jitter_s = 0.010`.
- Live capture sessions store `frames/`, `capture.json`, `calibration.json`, `result.json`, `grid_fit_debug.png`, `grid_composed.png`, `grid_debug.png`, `candidate_debug.png`, and `solve_order.txt` in `sessions/`.
- `replay --board-json` still produces exact website-board sessions for offline debugging, while `replay --images` and `replay --session` continue to use the local image reconstruction pipeline.
- Reconstruction now hard-rejects horizontal squeeze transition frames and only chooses crops from fully-open reveal plateaus.
- Group output now keeps a stable click order (top-left first, then row-major) and prints 3- and 4-match solve sequences to the console after reconstruction.
- The matching pipeline is intentionally conservative. Ambiguous candidate groups stay unresolved instead of being forced into clicks.
