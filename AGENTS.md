# Agent Handoff Notes

## Latest Changes (2025-10-31)

- Introduced dynamic column inference and override support inside the Streamlit sidebar.
- Added [`ColumnCandidates`](app/utils/data_io.py#L66) and [`ColumnSelection`](app/utils/data_io.py#L52) types to capture inferred defaults and user choices.
- Extended [`validate_data`](app/utils/data_io.py#L202) to honour explicit column selections while maintaining robust auto-detection.
- Updated [`prepare_dataset`](app/main.py#L63) to accept caller-provided `ColumnSelection` instances and align true labels using the resolved column name.
- Created new regression coverage in [`tests/test_data_io.py`](tests/test_data_io.py#L1) for wide/long selection paths and column inference.
- Refreshed [`README.md`](README.md#L63) documentation to describe the new Column Mapping workflow and testing instructions.

## Operational Checklist

1. Run the full suite before shipping changes:

   ```bash
   uv run --dev pytest -v
   ```

2. When modifying column handling:
   - Keep `ColumnCandidates` exhaustive enough to cover common customer schemas.
   - Update [`infer_column_candidates`](app/utils/data_io.py#L120) alongside the README examples.
   - Ensure any new defaults flow through [`column_selection`](app/main.py#L556) to maintain UI coherence.

3. For UI additions in the Column Mapping panel:
   - Store selections in `st.session_state` using the `column_mapping_*` prefix to leverage cache invalidation logic on signature changes.
   - Reset threshold-related session keys whenever the selection signature changes to avoid stale state.

4. If adding new exports or data artefacts:
   - Document them in the README exports section.
  - Provide regression coverage either in existing test modules or create a new targeted test file.


5. Type check using `ty`.

```bash
uv run ty check
```

## Known Considerations

- Long-format datasets without a reliable sample identifier fall back to synthetic row indices. Downstream alignment still works, but warn users in UI if you detect potential duplicates.
- Wide-format uploads with fewer than two score columns display an informational message; consider tightening this into a blocking validation if required.

## Future Opportunities

- Auto-persist the column selection signature so users reopening the app with the same schema skip manual remapping.
- Surface quick validation badges (e.g., ✅ Numeric / ⚠️ Missing) next to each column selector in the sidebar for richer feedback.
- Explore support for multi-file uploads (scores + metadata) by extending `ColumnSelection` to track auxiliary frames.

Keep this file updated after each significant change so future agents can land quickly.