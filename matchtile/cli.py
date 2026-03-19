from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from matchtile.autoplay import auto_click_groups
from matchtile.calibration_store import calibration_profile_path, load_calibration, save_calibration
from matchtile.capture import capture_frame, capture_frames, wait_for_hotkey
from matchtile.config import DEFAULT_CONFIG_PATH, ConfigLoadResult, MatchTileConfig
from matchtile.models import Calibration, ReconstructionResult
from matchtile.overlay import launch_overlay
from matchtile.phantom_board import (
    PHANTOM_BOARD_URL,
    board_source_metadata,
    build_reconstruction_from_phantom_board,
    load_phantom_board,
    load_phantom_metadata,
    metadata_source,
    save_phantom_board,
)
from matchtile.runtime_control import AbortRequested, StopToken
from matchtile.session import create_session_dir
from matchtile.vision import (
    build_manual_calibration,
    edit_calibration,
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
    calibrate.add_argument("--rows", type=int, default=None)
    calibrate.add_argument("--cols", type=int, default=None)
    calibrate.add_argument("--group-size", type=int, default=None)

    replay = subparsers.add_parser("replay", help="Reconstruct from a session directory, website board JSON, or image list.")
    replay.add_argument("--session", type=Path)
    replay.add_argument("--images", nargs="+", type=Path)
    replay.add_argument("--board-json", type=Path)
    replay.add_argument("--calibration", type=Path)
    replay.add_argument("--max-group-size", type=int, default=None)

    arm = subparsers.add_parser(
        "arm",
        help="Wait for a hotkey, capture the reveal, and solve using the saved local calibration.",
    )
    arm.add_argument("--window", type=str, default=None)
    arm.add_argument("--overlay", action="store_true")
    arm.add_argument("--start-now", action="store_true")
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


def _config_runtime_summary(config: MatchTileConfig, config_load: ConfigLoadResult) -> list[str]:
    if config_load.created_default_file:
        source = f"Config source: defaults from code (saved to {config_load.path})"
    else:
        source = f"Config source: {config_load.path}"
    effective = (
        "Effective runtime config: "
        f"capture_fps={config.capture_fps}, "
        f"reveal_duration_s={config.reveal_duration_s:.2f}, "
        f"max_group_size={config.max_group_size}"
    )
    return [source, effective]


def _format_group_click_order(group) -> str:
    if group.click_order_positions:
        return " -> ".join(f"r{item['row']:02d}c{item['col']:02d}" for item in group.click_order_positions)
    ordered = group.click_order or group.members
    return " -> ".join(ordered)


def _group_sort_key(group) -> tuple[int, int, str]:
    if group.click_order_positions:
        first = group.click_order_positions[0]
        return int(first["row"]), int(first["col"]), group.label
    first = (group.click_order or group.members or [group.label])[0]
    return 9999, 9999, first


def _write_solve_order(result: ReconstructionResult, output_dir: Path) -> Path:
    lines: list[str] = []
    ordered_groups = sorted(result.groups, key=_group_sort_key)
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
    ordered_groups = sorted(result.groups, key=_group_sort_key)
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
    raise RuntimeError("Replay with --images or --board-json requires --calibration <path>.")


def _load_live_calibration(config: MatchTileConfig, title_regex: str) -> tuple[Calibration, str, tuple[int, int], object]:
    target = find_window(title_regex)
    profile_path = calibration_profile_path(config, target.title, target.client_rect.width, target.client_rect.height)
    if not profile_path.exists():
        raise RuntimeError(
            "No saved calibration matched the current Discord window size. "
            f"Run `matchtile calibrate --window \"{target.title}\"` first."
        )
    calibration = load_calibration(profile_path)
    return calibration, str(profile_path), (target.client_rect.x, target.client_rect.y), target


def _resolve_replay_board(args: argparse.Namespace) -> tuple[str, object | None, Path | None]:
    if args.board_json:
        return "board-json", load_phantom_board(args.board_json), args.board_json
    if args.session and (args.session / "phantom_board.json").exists():
        board_path = args.session / "phantom_board.json"
        return "session-phantom", load_phantom_board(board_path), board_path
    return "legacy", None, None


def _session_board_metadata(session_dir: Path) -> tuple[dict | None, Path | None]:
    metadata_path = session_dir / "board_metadata.json"
    if metadata_path.exists():
        return load_phantom_metadata(metadata_path), metadata_path
    return None, None


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

    existing_calibration: Calibration | None = None
    if args.image:
        image = load_image(args.image)
        out_path = args.out or _image_calibration_path(args.image)
        window_title = None
        if out_path.exists():
            existing_calibration = load_calibration(out_path)
    else:
        title_regex = args.window or config.window_title_regex
        target = find_window(title_regex)
        bring_window_to_front(target.hwnd)
        image = capture_frame(target.client_rect)
        out_path = args.out or calibration_profile_path(config, target.title, target.client_rect.width, target.client_rect.height)
        window_title = target.title
        if out_path.exists():
            existing_calibration = load_calibration(out_path)

    rows = args.rows or (existing_calibration.rows if existing_calibration else None)
    cols = args.cols or (existing_calibration.cols if existing_calibration else None)
    group_size = args.group_size or (existing_calibration.group_size if existing_calibration else config.max_group_size)
    if rows is None or cols is None:
        raise RuntimeError("Fresh calibration requires --rows and --cols unless a saved profile already exists for this window size.")

    if existing_calibration:
        print(f"Loaded existing calibration from {out_path}")
    else:
        print("No existing calibration found. Starting fresh corner capture.")
    calibration = edit_calibration(
        image,
        initial_rows=rows,
        initial_cols=cols,
        initial_group_size=group_size,
        existing_calibration=existing_calibration,
    )
    save_calibration(out_path, calibration)
    print(f"Saved calibration to {out_path}")
    if window_title:
        print(f"Window: {window_title}")
        print(f"Client size: {calibration.client_width}x{calibration.client_height}")
    print(f"Board bounds: {calibration.board_rect}")
    print(f"Rows: {calibration.rows} | Cols: {calibration.cols} | Max group size: {calibration.group_size}")
    return 0


def cmd_replay(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    replay_mode, board, board_path = _resolve_replay_board(args)
    calibration = _load_replay_calibration(args, config)
    if replay_mode in {"board-json", "session-phantom"}:
        if args.max_group_size is not None and args.max_group_size != board.pairSize:
            raise RuntimeError(
                f"--max-group-size {args.max_group_size} did not match phantom board pairSize {board.pairSize}."
            )
        if replay_mode == "session-phantom" and args.session:
            session_dir = args.session
        else:
            session_dir = create_session_dir(config)
            save_phantom_board(session_dir / "phantom_board.json", board)
            save_calibration(session_dir / "calibration.json", calibration)
        source = board_source_metadata(board, str(board_path or PHANTOM_BOARD_URL), mode=replay_mode, fetched_at=datetime.now(timezone.utc))
        result = build_reconstruction_from_phantom_board(board, calibration, session_dir=session_dir, board_source=source)
        solve_order_path = _write_solve_order(result, session_dir)
        result.solve_order_path = str(solve_order_path)
        result.save(session_dir / "result.json")
        print(f"Reconstructed phantom board into {session_dir / 'result.json'}")
    elif args.session:
        metadata, metadata_path = _session_board_metadata(args.session)
        required_group_size = None
        max_group_size = calibration.group_size if calibration.group_size >= 2 else config.max_group_size
        if metadata is not None and calibration.source == "website-metadata":
            required_group_size = metadata["pair_size"]
            max_group_size = required_group_size
        if required_group_size is not None:
            if args.max_group_size is not None and args.max_group_size != required_group_size:
                raise RuntimeError(
                    f"--max-group-size {args.max_group_size} did not match session pair size {required_group_size}."
                )
        elif args.max_group_size is not None:
            max_group_size = max(2, min(args.max_group_size, 4))
        config.max_group_size = max_group_size
        result = reconstruct_from_session(
            args.session,
            config,
            calibration=calibration,
            required_group_size=required_group_size,
        )
        if metadata is not None:
            result.board_source = metadata_source(
                metadata,
                PHANTOM_BOARD_URL,
                mode="session-metadata",
                fetched_at=datetime.now(timezone.utc),
            )
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
    print(f"Calibration source: {result.calibration.source}")
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


def cmd_arm(args: argparse.Namespace, config: MatchTileConfig, config_load: ConfigLoadResult) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    title_regex = args.window or config.window_title_regex
    base_calibration, calibration_path, screen_offset, target = _load_live_calibration(config, title_regex)
    reveal_duration = config.reveal_duration_s
    click_delay_s = config.click_delay_s
    move_settle_s = config.move_settle_s
    mouse_down_hold_s = config.mouse_down_hold_s
    timing_jitter_s = config.timing_jitter_s
    bring_window_to_front(target.hwnd)
    session_dir = create_session_dir(config)
    print(f"Target window: {target.title}")
    for line in _config_runtime_summary(config, config_load):
        print(line)
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
    print(
        "Loaded calibration: "
        f"{base_calibration.cols} cols x {base_calibration.rows} rows | max group size {base_calibration.group_size}"
    )

    with StopToken() as stop_token:
        max_group_size = base_calibration.group_size
        if args.max_group_size is not None:
            if args.max_group_size > max_group_size:
                raise RuntimeError(
                    f"--max-group-size {args.max_group_size} exceeded calibration group size {max_group_size}."
                )
            max_group_size = max(2, min(args.max_group_size, 4))
        config.max_group_size = max_group_size
        save_calibration(session_dir / "calibration.json", base_calibration)

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
            __import__("json").dumps(
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
        print(f"Reconstructing board from local reveal frames with group sizes 2-{max_group_size}...")
        result = reconstruct_from_session(
            session_dir,
            config,
            calibration=base_calibration,
            required_group_size=None,
        )
        result.board_source = {
            "mode": "live-capture",
            "rows": base_calibration.rows,
            "cols": base_calibration.cols,
            "group_size": base_calibration.group_size,
            "max_group_size": max_group_size,
            "calibration_path": calibration_path,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        solve_order_path = _write_solve_order(result, session_dir)
        result.solve_order_path = str(solve_order_path)
        result.save(session_dir / "result.json")
        print(
            f"Captured {metadata.frame_count} frame(s) into {session_dir} "
            f"and reconstructed {len(result.groups)} group(s) up to size {max_group_size}."
        )
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
                + (
                    f" Click delay: {click_delay_s:.3f}s. "
                    f"Move settle: {move_settle_s:.3f}s. "
                    f"Hold: {mouse_down_hold_s:.3f}s. "
                    f"Jitter: +/-{timing_jitter_s:.3f}s."
                )
            )
            clicked = auto_click_groups(
                session_dir / "result.json",
                min_confidence=min_confidence,
                click_delay_s=click_delay_s,
                move_settle_s=move_settle_s,
                mouse_down_hold_s=mouse_down_hold_s,
                timing_jitter_s=timing_jitter_s,
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
    result_path = args.session / "result.json"
    result = ReconstructionResult.load(result_path)
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    else:
        config.max_group_size = result.calibration.group_size
    title_regex = args.window or config.window_title_regex
    screen_offset = _offset_for_overlay(result.calibration, title_regex)
    click_delay_s = config.click_delay_s
    move_settle_s = config.move_settle_s
    mouse_down_hold_s = config.mouse_down_hold_s
    timing_jitter_s = config.timing_jitter_s
    with StopToken() as stop_token:
        if args.autoplay:
            print(
                "Overlay autoplay timing: "
                f"delay {click_delay_s:.3f}s, "
                f"settle {move_settle_s:.3f}s, "
                f"hold {mouse_down_hold_s:.3f}s, "
                f"jitter +/-{timing_jitter_s:.3f}s."
            )
            clicked = auto_click_groups(
                result_path,
                min_confidence=args.autoplay_min_confidence or config.group_confidence_threshold,
                click_delay_s=click_delay_s,
                move_settle_s=move_settle_s,
                mouse_down_hold_s=mouse_down_hold_s,
                timing_jitter_s=timing_jitter_s,
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
        config_load = MatchTileConfig.load_with_result(args.config)
        config = config_load.config

        if args.command == "calibrate":
            return cmd_calibrate(args, config)
        if args.command == "replay":
            return cmd_replay(args, config)
        if args.command == "arm":
            return cmd_arm(args, config, config_load)
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
