# Match Tile Assistant

`matchtile` is a Windows-first assistant for the Discord Match Tile activity.
It now uses explicit manual calibration instead of automatic board detection:

- calibrate the playfield once by clicking its 4 corners
- save that calibration by Discord client size
- fetch the solved live board from the Cryo Archive site
- derive exact groups from the website's `imgIdx` data
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
matchtile calibrate --window "NULL//TRANSMIT.ERR"
matchtile calibrate --image empty-board-2.png
matchtile replay --session sessions\20260319-024228
matchtile replay --board-json tmp_live_board.json --calibration calibrations\NULL_TRANSMIT.ERR_2556x1360.calibration.json
matchtile replay --images "Screenshot 2026-03-19 005448.png" --calibration empty-board-2.calibration.json
matchtile arm --window "NULL//TRANSMIT.ERR" --overlay
matchtile arm --window "NULL//TRANSMIT.ERR" --max-group-size 4 --autoplay
matchtile overlay --session sessions\20260319-013000 --window "NULL//TRANSMIT.ERR"
```

## Notes

- `calibrate` is the only way to define board placement. It asks you to click top-left, top-right, bottom-right, and bottom-left, then saves the playfield corners for that Discord client size.
- Press `Enter` or `Space` to save after the fourth corner. Press `Backspace` to undo the last corner, `R` to reset, or `Esc` to cancel.
- Live window calibrations are saved under `calibrations/` and reused automatically by `arm` for the same Discord client size.
- `arm` no longer auto-calibrates. If no matching calibration exists, it stops and tells you to run `matchtile calibrate` first.
- `arm` waits for `F8` by default, polls `https://marathon.winnower.garden/api/cryo-phantom-board` for a fresh live board, builds exact groups from `imgIdx`, and can launch the overlay immediately.
- `replay --board-json` requires `--calibration <path>`. `replay --session` uses `session\calibration.json` when present and automatically switches to website-board replay if `session\phantom_board.json` exists.
- Use `--max-group-size <2-4>` on `replay`, `arm`, or `overlay` to cap inferred match-group size for a specific run.
- `F12` is a global emergency stop for `arm`, overlay mode, and auto-click mode.
- `arm --autoplay` clicks only high-confidence match groups by default and aborts if the post-click verification does not detect the expected board change.
- Autoplay now uses a move-settle and mouse-hold click primitive by default: `click_delay_s = 0.25`, `move_settle_s = 0.035`, `mouse_down_hold_s = 0.045`, `timing_jitter_s = 0.010`.
- Website-backed sessions store `phantom_board.json`, `calibration.json`, `result.json`, `grid_fit_debug.png`, `grid_composed.png`, `grid_debug.png`, and `solve_order.txt` in `sessions/`.
- Legacy capture/image reconstruction still works for old sessions and still writes `candidate_debug.png` when transition-frame analysis is involved.
- Reconstruction now hard-rejects horizontal squeeze transition frames and only chooses crops from fully-open reveal plateaus.
- Group output now keeps a stable click order (top-left first, then row-major) and prints 3- and 4-match solve sequences to the console after reconstruction.
- The matching pipeline is intentionally conservative. Ambiguous candidate groups stay unresolved instead of being forced into clicks.
