# Match Tile Assistant

`matchtile` is a Windows-first assistant for the Discord Match Tile activity.
It now uses explicit manual calibration instead of automatic board detection:

- calibrate the playfield once by clicking its 4 corners
- save that calibration by Discord client size
- capture the reveal burst
- reconstruct which face belongs to each hidden tile
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
matchtile calibrate --window "NULL//TRANSMIT.ERR" --col 64 --row 25
matchtile calibrate --image empty-board-2.png --col 64 --row 25
matchtile replay --session sessions\20260319-024228
matchtile replay --images "Screenshot 2026-03-19 005448.png" --calibration empty-board-2.calibration.json
matchtile arm --window "NULL//TRANSMIT.ERR" --overlay
matchtile arm --window "NULL//TRANSMIT.ERR" --reveal-duration 10 --overlay
matchtile arm --window "NULL//TRANSMIT.ERR" --max-group-size 4 --autoplay
matchtile overlay --session sessions\20260319-013000 --window "NULL//TRANSMIT.ERR"
```

## Notes

- `calibrate` is the only way to define board placement. It requires `--col` and `--row`, then asks you to click top-left, top-right, bottom-right, and bottom-left.
- After the fourth click, `calibrate` immediately projects the grid in the same window. Use the arrow keys to adjust counts live: `Left` / `Right` decrease or increase columns, and `Up` / `Down` increase or decrease rows.
- Press `Enter` or `Space` to save the currently projected grid. Press `Backspace` to undo the last corner, `R` to reset, or `Esc` to cancel.
- Live window calibrations are saved under `calibrations/` and reused automatically by `arm` for the same Discord client size.
- `arm` no longer auto-calibrates. If no matching calibration exists, it stops and tells you to run `matchtile calibrate` first.
- `replay --images` requires `--calibration <path>`. `replay --session` uses `session\calibration.json` when present.
- `arm` waits for `F8` by default, captures the configured reveal burst, reconstructs the board, and can launch the overlay immediately.
- Use `--reveal-duration <seconds>` on `arm` to override the capture window for a specific run; otherwise it uses `reveal_duration_s` from `matchtile.json`.
- Use `--max-group-size <2-4>` on `replay`, `arm`, or `overlay` to cap inferred match-group size for a specific run.
- `F12` is a global emergency stop for `arm`, overlay mode, and auto-click mode.
- `arm --autoplay` clicks only high-confidence match groups by default and aborts if the post-click verification does not detect the expected board change.
- Every session stores raw frames, `calibration.json`, and reconstruction JSON in `sessions/`.
- Every solve also writes `grid_fit_debug.png`, `grid_composed.png`, and `grid_debug.png` into the session directory for debugging.
- The matching pipeline is intentionally conservative. Ambiguous groups stay visible with lower confidence instead of being hidden.
