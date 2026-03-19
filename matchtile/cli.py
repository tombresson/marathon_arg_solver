from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from matchtile.autoplay import auto_click_groups
from matchtile.capture import capture_frame, capture_frames, wait_for_hotkey
from matchtile.config import DEFAULT_CONFIG_PATH, MatchTileConfig
from matchtile.models import Rect
from matchtile.overlay import launch_overlay
from matchtile.runtime_control import AbortRequested, StopToken
from matchtile.session import create_session_dir
from matchtile.vision import calibrate_grid, crop_rect, detect_board_rect, detect_discord_activity_region, load_image, manual_select_rect, reconstruct_from_images, reconstruct_from_session
from matchtile.windowing import bring_window_to_front, find_window


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matchtile", description="Discord Match Tile assistant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="Detect the board and save calibration info.")
    calibrate.add_argument("--image", type=Path)
    calibrate.add_argument("--window", type=str, default=None)
    calibrate.add_argument("--full-window", action="store_true")
    calibrate.add_argument("--out", type=Path)
    calibrate.add_argument("--manual-board", action="store_true")

    replay = subparsers.add_parser("replay", help="Reconstruct from a session directory or image list.")
    replay.add_argument("--session", type=Path)
    replay.add_argument("--images", nargs="+", type=Path)
    replay.add_argument("--manual-board", action="store_true")
    replay.add_argument("--max-group-size", type=int, default=None)

    arm = subparsers.add_parser("arm", help="Wait for a hotkey, capture the reveal burst, and reconstruct.")
    arm.add_argument("--window", type=str, default=None)
    arm.add_argument("--overlay", action="store_true")
    arm.add_argument("--start-now", action="store_true")
    arm.add_argument("--manual-board", action="store_true")
    arm.add_argument("--reveal-duration", type=float, default=None)
    arm.add_argument("--max-group-size", type=int, default=None)
    arm.add_argument("--autoplay", action="store_true")
    arm.add_argument("--autoplay-min-confidence", type=float, default=None)
    arm.add_argument("--include-ambiguous", action="store_true")

    overlay = subparsers.add_parser("overlay", help="Display the transparent overlay for a session result.")
    overlay.add_argument("--session", type=Path, required=True)
    overlay.add_argument("--max-group-size", type=int, default=None)
    overlay.add_argument("--autoplay", action="store_true")
    overlay.add_argument("--autoplay-min-confidence", type=float, default=None)
    overlay.add_argument("--include-ambiguous", action="store_true")
    return parser


def _default_calibration_path(window_title: str | None = None) -> Path:
    if not window_title:
        return Path("live.calibration.json")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", window_title).strip("_")
    if not safe:
        safe = "live"
    return Path(f"{safe}.calibration.json")


def cmd_calibrate(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.image:
        image = load_image(args.image)
        out_path = args.out or args.image.with_suffix(".calibration.json")
        window_title = None
    else:
        title_regex = args.window or config.window_title_regex
        target = find_window(title_regex)
        bring_window_to_front(target.hwnd)
        image = capture_frame(target.client_rect)
        out_path = args.out or _default_calibration_path(target.title)
        window_title = target.title

    if args.manual_board:
        board_rect = manual_select_rect(image)
        board_image = crop_rect(image, board_rect)
    elif args.full_window:
        activity = detect_discord_activity_region(image)
        board = detect_board_rect(crop_rect(image, activity), restrict_left_bias=True)
        board_rect = Rect(activity.x + board.x, activity.y + board.y, board.width, board.height)
        board_image = crop_rect(image, board_rect)
    else:
        board_rect = detect_board_rect(image)
        board_image = crop_rect(image, board_rect)
    calibration = calibrate_grid(board_image, board_rect)
    out_path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved calibration to {out_path}")
    if window_title:
        print(f"Window: {window_title}")
    print(f"Board rect: {board_rect}")
    print(f"Grid estimate: {calibration.rows} rows x {calibration.cols} cols")
    return 0


def cmd_replay(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    board_override = None
    if args.session:
        if args.manual_board:
            first_frame = sorted((args.session / "frames").glob("*.png"))[0]
            board_override = manual_select_rect(load_image(first_frame))
        result = reconstruct_from_session(args.session, config, board_rect_override=board_override)
        print(f"Reconstructed session into {args.session / 'result.json'}")
    else:
        session_dir = create_session_dir(config)
        if args.manual_board and args.images:
            board_override = manual_select_rect(load_image(args.images[0]))
        result = reconstruct_from_images(args.images or [], session_dir=session_dir, config=config, board_rect_override=board_override)
        result.save(session_dir / "result.json")
        print(f"Reconstructed images into {session_dir / 'result.json'}")
    print(f"Groups: {len(result.groups)} | Unresolved: {len(result.unresolved)}")
    if result.grid_fit_debug_path:
        print(f"Grid fit debug: {result.grid_fit_debug_path}")
    if result.grid_composed_path:
        print(f"Composed grid: {result.grid_composed_path}")
    if result.grid_debug_path:
        print(f"Debug grid: {result.grid_debug_path}")
    return 0


def cmd_arm(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    title_regex = args.window or config.window_title_regex
    reveal_duration = args.reveal_duration if args.reveal_duration is not None else config.reveal_duration_s
    click_delay_s = 0.75
    target = find_window(title_regex)
    bring_window_to_front(target.hwnd)
    session_dir = create_session_dir(config)
    print(f"Target window: {target.title}")
    print(
        "Capture region: "
        f"{target.client_rect.width}x{target.client_rect.height} at "
        f"({target.client_rect.x}, {target.client_rect.y})"
    )
    print(f"Session directory: {session_dir}")
    print(
        "Capture settings: "
        f"{config.capture_fps} FPS for {reveal_duration:.2f}s "
        f"(about {int(config.capture_fps * reveal_duration)} frames expected)"
    )
    with StopToken() as stop_token:
        if not args.start_now:
            wait_for_hotkey(stop_token=stop_token)
            print("Activating target window for capture.")
            bring_window_to_front(target.hwnd)
        else:
            print("Start-now enabled. Beginning reveal capture immediately.")
            bring_window_to_front(target.hwnd)
        print("Capturing reveal frames...")
        metadata = capture_frames(
            target.client_rect,
            fps=config.capture_fps,
            duration_s=reveal_duration,
            out_dir=session_dir,
            stop_token=stop_token,
        )
        print(f"Capture complete. Saved {metadata.frame_count} frame(s).")
        (session_dir / "capture.json").write_text(
            json.dumps(
                {
                    "fps": metadata.fps,
                    "duration_s": metadata.duration_s,
                    "frame_count": metadata.frame_count,
                    "capture_rect": metadata.capture_rect.as_dict(),
                    "window_title": target.title,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        board_override = None
        if args.manual_board:
            first_frame = sorted((session_dir / "frames").glob("*.png"))[0]
            print("Manual board mode enabled. Select the board region from the first captured frame.")
            board_override = manual_select_rect(load_image(first_frame))
        print("Reconstructing board and matching groups...")
        result = reconstruct_from_session(session_dir, config, board_rect_override=board_override)
        result.calibration.board_rect.x += target.client_rect.x
        result.calibration.board_rect.y += target.client_rect.y
        for center in result.calibration.centers:
            center.x += target.client_rect.x
            center.y += target.client_rect.y
        result.save(session_dir / "result.json")
        print(f"Captured {metadata.frame_count} frames into {session_dir}")
        print(f"Groups: {len(result.groups)} | Unresolved: {len(result.unresolved)}")
        if result.grid_fit_debug_path:
            print(f"Grid fit debug: {result.grid_fit_debug_path}")
        if result.grid_composed_path:
            print(f"Composed grid: {result.grid_composed_path}")
        if result.grid_debug_path:
            print(f"Debug grid: {result.grid_debug_path}")
        min_confidence = args.autoplay_min_confidence or config.group_confidence_threshold
        if args.autoplay:
            print(
                "Autoplay enabled. "
                f"Clicking confident groups with threshold {min_confidence:.2f}"
                + (" including ambiguous groups." if args.include_ambiguous else ".")
                + f" Click delay: {click_delay_s:.2f}s."
            )
            clicked = auto_click_groups(
                session_dir / "result.json",
                min_confidence=min_confidence,
                click_delay_s=click_delay_s,
                verify=True,
                include_ambiguous=args.include_ambiguous,
                capture_rect=(target.client_rect.x, target.client_rect.y, target.client_rect.width, target.client_rect.height),
                stop_token=stop_token,
            )
            print(f"Auto-clicked {clicked} tile(s) across confident groups.")
        if args.overlay:
            rect = target.client_rect
            print("Launching overlay...")
            return launch_overlay(
                session_dir / "result.json",
                config,
                live_capture_rect=(rect.x, rect.y, rect.width, rect.height),
                stop_token=stop_token,
            )
    return 0


def cmd_overlay(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    result_path = args.session / "result.json"
    click_delay_s = 0.75
    with StopToken() as stop_token:
        if args.autoplay:
            clicked = auto_click_groups(
                result_path,
                min_confidence=args.autoplay_min_confidence or config.group_confidence_threshold,
                click_delay_s=click_delay_s,
                verify=False,
                include_ambiguous=args.include_ambiguous,
                stop_token=stop_token,
            )
            print(f"Auto-clicked {clicked} tile(s) across confident groups.")
        return launch_overlay(result_path, config, stop_token=stop_token)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        config = MatchTileConfig.load(args.config)

        if args.command == "calibrate":
            return cmd_calibrate(args, config)
        if args.command == "replay":
            return cmd_replay(args, config)
        if args.command == "arm":
            return cmd_arm(args, config)
        if args.command == "overlay":
            return cmd_overlay(args, config)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except AbortRequested as exc:
        print(str(exc))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
