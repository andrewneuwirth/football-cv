# Examples

## `positions.sample.json`

A sample of the `positions.json` artifact written by the final pipeline stage
(`python -m ml.cli positions <play_id> --data ./data`).

It maps each tracked player's track id to a generic position token:

```json
{ "1": "OL", "2": "WR", "3": "QB", "4": "RB", "11": "DL", "12": "LB", "13": "CB", "14": "S" }
```

Track ids `1–4` are offensive players (`OL`, `WR`, `QB`, `RB`); `11–14` are
defensive players (`DL`, `LB`, `CB`, `S`). Tokens come from the taxonomy
documented in the top-level [README](../README.md#position-taxonomy).
