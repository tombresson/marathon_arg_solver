from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from matchtile.autoplay import auto_click_groups
from matchtile.calibration_store import calibration_profile_path, load_calibration, save_calibration
from matchtile.capture import capture_frame, capture_frames, wait_for_hotkey
from matchtile.config import DEFAULT_CONFIG_PATH, MatchTileConfig
from matchtile.models import Calibration, ReconstructionResult
from matchtile.overlay import launch_overlay
from matchtile.runtime_control import AbortRequested, StopToken
from matchtile.session import create_session_dir
from matchtile.vision import (
    build_manual_calibration,
    load_image,
    manual_select_corners,
    reconstruct_from_images,
    reconstruct_from_session,
)
from matchtile.windowing import bring_window_to_front, find_window


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matchtile", description="Discord Match Tile assistant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="Create a manual 4-corner calibration profile.")
    calibrate.add_argument("--image", type=Path)
    calibrate.add_argument("--window", type=str, default=None)
    calibrate.add_argument("--out", type=Path)
    calibrate.add_argument("--col", type=int, required=True)
    calibrate.add_argument("--row", type=int, required=True)

    replay = subparsers.add_parser("replay", help="Reconstruct from a session directory or image list.")
    replay.add_argument("--session", type=Path)
    replay.add_argument("--images", nargs="+", type=Path)
    replay.add_argument("--calibration", type=Path)
    replay.add_argument("--max-group-size", type=int, default=None)

    arm = subparsers.add_parser("arm", help="Wait for a hotkey, capture the reveal burst, and reconstruct.")
    arm.add_argument("--window", type=str, default=None)
    arm.add_argument("--overlay", action="store_true")
    arm.add_argument("--start-now", action="store_true")
    arm.add_argument("--reveal-duration", type=float, default=None)
    arm.add_argument("--max-group-size", type=int, default=None)
    arm.add_argument("--autoplay", action="store_true")
    arm.add_argument("--autoplay-min-confidence", type=float, default=None)
    arm.add_argument("--include-ambiguous", action="store_true")

    overlay = subparsers.add_parser("overlay", help="Display the transparent overlay for a session result.")
    overlay.add_argument("--session", type=Path, required=True)
    overlay.add_argument("--window", type=str, default=None)
    overlay.add_argument("--max-group-size", type=int, default=None)
    overlay.add_argument("--autoplay", action="store_true")
    overlay.add_argument("--autoplay-min-confidence", type=float, default=None)
    overlay.add_argument("--include-ambiguous", action="store_true")
    return parser


def _image_calibration_path(image_path: Path) -> Path:
    return image_path.with_suffix(".calibration.json")


def _format_group_click_order(group) -> str:
    if group.click_order_positions:
        return " -> ".join(f"r{item['row']:02d}c{item['col']:02d}" for item in group.click_order_positions)
    ordered = group.click_order or group.members
    return " -> ".join(ordered)


def _write_solve_order(result: ReconstructionResult, output_dir: Path) -> Path:
    lines: list[str] = []
    ordered_groups = sorted(result.groups, key=lambda group: (-group.group_size, -group.confidence, group.label))
    for group in ordered_groups:
        lines.append(
            f"{group.label} size {group.group_size} conf {group.confidence:.3f} "
            f"sim_min {group.similarity_min:.3f} sim_mean {group.similarity_mean:.3f}: "
            f"{_format_group_click_order(group)}"
        )
    output_path = output_dir / "solve_order.txt"
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def _print_group_summary(result: ReconstructionResult) -> None:
    ordered_groups = sorted(result.groups, key=lambda group: (-group.group_size, -group.confidence, group.label))
    interesting = [group for group in ordered_groups if group.group_size >= 3]
    if not interesting:
        interesting = ordered_groups[:10]
    for group in interesting:
        print(f"{group.label} size {group.group_size}: {_format_group_click_order(group)}")


def _load_replay_calibration(args: argparse.Namespace, config: MatchTileConfig) -> Calibration:
    if args.calibration:
        return load_calibration(args.calibration)
    if args.session:
        session_calibration = args.session / "calibration.json"
        if session_calibration.exists():
            return load_calibration(session_calibration)
        raise RuntimeError(
            f"No calibration found for {args.session}. "
            "Pass --calibration <path> or replay a session created by `matchtile arm`."
        )
    raise RuntimeError("Replay with --images requires --calibration <path>.")


def _load_live_calibration(config: MatchTileConfig, title_regex: str) -> tuple[Calibration, str, tuple[int, int], object]:
    target = find_window(title_regex)
    profile_path = calibration_profile_path(config, target.title, target.client_rect.width, target.client_rect.height)
    if not profile_path.exists():
        raise RuntimeError(
            "No saved calibration matched the current Discord window size. "
            f"Run `matchtile calibrate --window \"{target.title}\" --col <cols> --row <rows>` first."
        )
    calibration = load_calibration(profile_path)
    return calibration, str(profile_path), (target.client_rect.x, target.client_rect.y), target


def _offset_for_overlay(calibration: Calibration, title_regex: str) -> tuple[int, int]:
    try:
        target = find_window(title_regex)
    except RuntimeError:
        return (0, 0)
    if calibration.client_width and calibration.client_height:
        if target.client_rect.width != calibration.client_width or target.client_rect.height != calibration.client_height:
            return (0, 0)
    return (target.client_rect.x, target.client_rect.y)


def cmd_calibrate(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if bool(args.image) == bool(args.window):
        raise RuntimeError("Calibration requires exactly one of --image or --window.")
    if args.col <= 0 or args.row <= 0:
        raise RuntimeError("--col and --row must both be positive integers.")

    if args.image:
        image = load_image(args.image)
        out_path = args.out or _image_calibration_path(args.image)
        window_title = None
    else:
        title_regex = args.window or config.window_title_regex
        target = find_window(title_regex)
        bring_window_to_front(target.hwnd)
        image = capture_frame(target.client_rect)
        out_path = args.out or calibration_profile_path(config, target.title, target.client_rect.width, target.client_rect.height)
        window_title = target.title

    print(f"Calibration target: {args.col} cols x {args.row} rows")
    print("Click top-left, top-right, bottom-right, then bottom-left.")
    corners, final_rows, final_cols = manual_select_corners(image, rows=args.row, cols=args.col)
    calibration = build_manual_calibration((image.shape[1], image.shape[0]), rows=final_rows, cols=final_cols, corners=corners)
    save_calibration(out_path, calibration)
    print(f"Saved calibration to {out_path}")
    if window_title:
        print(f"Window: {window_title}")
        print(f"Client size: {calibration.client_width}x{calibration.client_height}")
    print(f"Board bounds: {calibration.board_rect}")
    return 0


def cmd_replay(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    calibration = _load_replay_calibration(args, config)
    if args.session:
        result = reconstruct_from_session(args.session, config, calibration=calibration)
        solve_order_path = _write_solve_order(result, args.session)
        result.solve_order_path = str(solve_order_path)
        result.save(args.session / "result.json")
        print(f"Reconstructed session into {args.session / 'result.json'}")
    else:
        session_dir = create_session_dir(config)
        result = reconstruct_from_images(args.images or [], session_dir=session_dir, config=config, calibration=calibration)
        save_calibration(session_dir / "calibration.json", calibration)
        solve_order_path = _write_solve_order(result, session_dir)
        result.solve_order_path = str(solve_order_path)
        result.save(session_dir / "result.json")
        print(f"Reconstructed images into {session_dir / 'result.json'}")
    print(f"Calibration source: {calibration.source}")
    print(f"Groups: {len(result.groups)} | Unresolved: {len(result.unresolved)}")
    _print_group_summary(result)
    if result.grid_fit_debug_path:
        print(f"Grid fit debug: {result.grid_fit_debug_path}")
    if result.grid_composed_path:
        print(f"Composed grid: {result.grid_composed_path}")
    if result.grid_debug_path:
        print(f"Debug grid: {result.grid_debug_path}")
    if result.candidate_debug_path:
        print(f"Candidate debug: {result.candidate_debug_path}")
    if result.solve_order_path:
        print(f"Solve order: {result.solve_order_path}")
    return 0


def cmd_arm(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    title_regex = args.window or config.window_title_regex
    calibration, calibration_path, screen_offset, target = _load_live_calibration(config, title_regex)
    reveal_duration = args.reveal_duration if args.reveal_duration is not None else config.reveal_duration_s
    click_delay_s = 0.75
    bring_window_to_front(target.hwnd)
    session_dir = create_session_dir(config)
    print(f"Target window: {target.title}")
    print(f"Calibration: {calibration_path}")
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
                    "calibration_path": calibration_path,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_calibration(session_dir / "calibration.json", calibration)

        print("Reconstructing board and matching groups...")
        result = reconstruct_from_session(session_dir, config, calibration=calibration)
        solve_order_path = _write_solve_order(result, session_dir)
        result.solve_order_path = str(solve_order_path)
        result.save(session_dir / "result.json")
        print(f"Captured {metadata.frame_count} frames into {session_dir}")
        print(f"Groups: {len(result.groups)} | Unresolved: {len(result.unresolved)}")
        _print_group_summary(result)
        if result.grid_fit_debug_path:
            print(f"Grid fit debug: {result.grid_fit_debug_path}")
        if result.grid_composed_path:
            print(f"Composed grid: {result.grid_composed_path}")
        if result.grid_debug_path:
            print(f"Debug grid: {result.grid_debug_path}")
        if result.candidate_debug_path:
            print(f"Candidate debug: {result.candidate_debug_path}")
        if result.solve_order_path:
            print(f"Solve order: {result.solve_order_path}")

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
                screen_offset=screen_offset,
                stop_token=stop_token,
            )
            print(f"Auto-clicked {clicked} tile(s) across confident groups.")
        if args.overlay:
            print("Launching overlay...")
            return launch_overlay(
                session_dir / "result.json",
                config,
                live_capture_rect=(target.client_rect.x, target.client_rect.y, target.client_rect.width, target.client_rect.height),
                screen_offset=screen_offset,
                stop_token=stop_token,
            )
    return 0


def cmd_overlay(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    result_path = args.session / "result.json"
    result = ReconstructionResult.load(result_path)
    title_regex = args.window or config.window_title_regex
    screen_offset = _offset_for_overlay(result.calibration, title_regex)
    click_delay_s = 0.75
    with StopToken() as stop_token:
        if args.autoplay:
            clicked = auto_click_groups(
                result_path,
                min_confidence=args.autoplay_min_confidence or config.group_confidence_threshold,
                click_delay_s=click_delay_s,
                verify=False,
                include_ambiguous=args.include_ambiguous,
                screen_offset=screen_offset,
                stop_token=stop_token,
            )
            print(f"Auto-clicked {clicked} tile(s) across confident groups.")
        return launch_overlay(result_path, config, screen_offset=screen_offset, stop_token=stop_token)


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
