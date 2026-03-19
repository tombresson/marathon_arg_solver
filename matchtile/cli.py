from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from matchtile.autoplay import auto_click_groups
from matchtile.calibration_store import calibration_profile_path, load_calibration, save_calibration
from matchtile.capture import capture_frame, wait_for_hotkey
from matchtile.config import DEFAULT_CONFIG_PATH, MatchTileConfig
from matchtile.models import Calibration, ReconstructionResult
from matchtile.overlay import launch_overlay
from matchtile.phantom_board import (
    PHANTOM_BOARD_URL,
    board_source_metadata,
    build_reconstruction_from_phantom_board,
    fetch_phantom_board,
    load_phantom_board,
    save_phantom_board,
    wait_for_fresh_phantom_board,
)
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

    replay = subparsers.add_parser("replay", help="Reconstruct from a session directory, website board JSON, or image list.")
    replay.add_argument("--session", type=Path)
    replay.add_argument("--images", nargs="+", type=Path)
    replay.add_argument("--board-json", type=Path)
    replay.add_argument("--calibration", type=Path)
    replay.add_argument("--max-group-size", type=int, default=None)

    arm = subparsers.add_parser("arm", help="Wait for a hotkey, fetch the live phantom board, and solve.")
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

    print("Click top-left, top-right, bottom-right, then bottom-left.")
    corners = manual_select_corners(image)
    calibration = build_manual_calibration((image.shape[1], image.shape[0]), rows=1, cols=1, corners=corners)
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


def cmd_arm(args: argparse.Namespace, config: MatchTileConfig) -> int:
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    title_regex = args.window or config.window_title_regex
    base_calibration, calibration_path, screen_offset, target = _load_live_calibration(config, title_regex)
    click_delay_s = config.click_delay_s
    move_settle_s = config.move_settle_s
    mouse_down_hold_s = config.mouse_down_hold_s
    timing_jitter_s = config.timing_jitter_s
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

    with StopToken() as stop_token:
        print(f"Fetching baseline phantom board from {PHANTOM_BOARD_URL}...")
        baseline_board = fetch_phantom_board()
        print(
            "Baseline board: "
            f"{baseline_board.width}x{baseline_board.height}, "
            f"pairSize {baseline_board.pairSize}, "
            f"startedAt {baseline_board.startedAt or 'unknown'}"
        )
        if not args.start_now:
            wait_for_hotkey(stop_token=stop_token)
            print("Activating target window for solve.")
            bring_window_to_front(target.hwnd)
        else:
            print("Start-now enabled. Polling for a fresh phantom board immediately.")
            bring_window_to_front(target.hwnd)

        print("Polling for a fresh phantom board...")
        board = wait_for_fresh_phantom_board(
            baseline_board,
            url=PHANTOM_BOARD_URL,
            poll_interval_s=0.25,
            timeout_s=10.0,
            stop_token=stop_token,
        )
        print(
            "Fresh board detected: "
            f"{board.width}x{board.height}, "
            f"pairSize {board.pairSize}, "
            f"startedAt {board.startedAt or 'unknown'}"
        )
        if args.max_group_size is not None and args.max_group_size != board.pairSize:
            raise RuntimeError(
                f"--max-group-size {args.max_group_size} did not match phantom board pairSize {board.pairSize}."
            )
        board_source = board_source_metadata(board, PHANTOM_BOARD_URL, mode="live-api", fetched_at=datetime.now(timezone.utc))
        save_phantom_board(session_dir / "phantom_board.json", board)

        print("Building exact groups from phantom board data...")
        result = build_reconstruction_from_phantom_board(board, base_calibration, session_dir=session_dir, board_source=board_source)
        save_calibration(session_dir / "calibration.json", result.calibration)
        solve_order_path = _write_solve_order(result, session_dir)
        result.solve_order_path = str(solve_order_path)
        result.save(session_dir / "result.json")
        print(f"Saved phantom board session into {session_dir}")
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
    if args.max_group_size is not None:
        config.max_group_size = max(2, min(args.max_group_size, 4))
    result_path = args.session / "result.json"
    result = ReconstructionResult.load(result_path)
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
