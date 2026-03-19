# Generalize Match Groups To 2-4 Members

## Summary
Update the solver, overlay, and autoplay pipeline so a tile identity can require 2, 3, or 4 matching cells within the same round.  
The tool should infer group size per identity, support mixed group sizes in one board, and only autoplay groups whose size and membership are both high-confidence. For 3/4-match groups, autoplay must click all members consecutively as one transaction.

## Key Changes
- Replace the pair-only solver model with a generic match-group model.
  Rename the conceptual output from “pairs” to “groups” and allow `members` length 2-4.
- Add a configurable maximum group size.
  Public interface: `max_group_size` in config, default `4`; optional CLI override on `arm` and `overlay` if desired.
- Rework grouping in the vision pipeline.
  Build a similarity graph over revealed cells, then infer candidate groups of size 2-4 instead of only mutual-best pairs.
  Score each candidate on:
  - internal similarity consistency
  - separation from nearby competing candidates
  - confidence in inferred group size
- Enforce conservative autoplay.
  Only click groups where:
  - inferred size is confident
  - all required members are present
  - no competing overlapping group has similar score
  For size 3/4 groups, click all members in order with existing delay and post-group verification.
- Update overlay semantics.
  Show group IDs instead of pair IDs, tint by confidence, and make group size visible in the HUD or label so 2/3/4 matches are distinguishable.
- Update solved-state verification.
  Do not assume exactly two changed cells; verify that all clicked members changed appropriately before marking the group solved.

## Interfaces And Behavior
- Config additions:
  - `max_group_size: int = 4`
  - optionally `group_confidence_threshold` if kept distinct from current pair threshold
- Data model changes:
  - replace `PairMatch` with a generic `MatchGroup` carrying `label`, `members`, `confidence`, `ambiguous`, and inferred `group_size`
  - rename `pairs` collections in result payloads to `groups`
- CLI/runtime messaging:
  - replace “Pairs” with “Groups”
  - autoplay logs should report the group size being clicked
- Autoplay policy:
  - never click partial hypotheses for 3/4 groups
  - skip ambiguous overlapping groups unless explicitly allowed later
  - preserve `F12` abort behavior during multi-click groups

## Test Plan
- Unit test grouping logic on synthetic descriptors for:
  - clean 2-member groups
  - clean 3-member groups
  - clean 4-member groups
  - mixed 2/3/4 groups in one round
  - overlapping/ambiguous candidates that should stay unresolved
- Replay test on saved sessions:
  - result payload contains groups with mixed sizes
  - unresolved count stays nonzero when size inference is uncertain instead of forcing wrong groups
- Autoplay tests:
  - 3/4-member groups are clicked in one uninterrupted sequence
  - verification requires all clicked members to change
  - abort path stops mid-run cleanly with `F12`
- Overlay tests:
  - labels render correctly for groups larger than 2
  - solved groups disappear only after full-group verification

## Assumptions
- Group size may vary by identity within the same round.
- Valid match sizes for v1 of this change are 2, 3, and 4 only.
- A correct 3/4-match requires clicking all members consecutively.
- Autoplay should remain conservative and only act on confident complete groups; uncertain groups stay visible for manual handling.
