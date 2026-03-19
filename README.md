# Match Tile Assistant

`matchtile` is a Windows-first assistant for the Discord Match Tile activity.
It focuses on a cautious workflow:

- detect the Discord activity area
- calibrate the tile grid for the current round
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
matchtile calibrate --window "NULL//TRANSMIT.ERR" --full-window
matchtile replay --images "Screenshot 2026-03-19 005448.png" "Screenshot 2026-03-19 010709.png"
matchtile arm --window "NULL//TRANSMIT.ERR" --overlay
matchtile arm --window "NULL//TRANSMIT.ERR" --reveal-duration 10 --overlay
matchtile arm --window "NULL//TRANSMIT.ERR" --max-group-size 4 --autoplay
matchtile arm --window "NULL//TRANSMIT.ERR" --autoplay
matchtile overlay --session sessions\20260319-013000
```

## Notes

- `arm` waits for `F8` by default, captures the configured reveal burst, reconstructs the board, and can launch the overlay immediately.
- Use `--reveal-duration <seconds>` on `arm` to override the capture window for a specific run; otherwise it uses `reveal_duration_s` from `matchtile.json`.
- Use `--max-group-size <2-4>` on `replay`, `arm`, or `overlay` to cap inferred match-group size for a specific run.
- `calibrate` can now grab the live Discord window directly; `--image` is still available for offline tuning.
- `F12` is a global emergency stop for `arm`, overlay mode, and auto-click mode.
- `arm --autoplay` clicks only high-confidence match groups by default and aborts if the post-click verification does not detect the expected board change.
- Every session stores raw frames, calibration output, and reconstruction JSON in `sessions/`.
- Every solve also writes `grid_composed.png` and `grid_debug.png` into the session directory for board reconstruction debugging.
- The matching pipeline is intentionally conservative. Ambiguous groups stay visible with lower confidence instead of being hidden.
