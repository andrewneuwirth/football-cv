# Examples

## `positions.sample.json`

A sample of the `positions.json` artifact written by the final pipeline stage
(`python -m footballcv.cli positions <play_id> --data ./data`).

It maps each classified defender's track id to a standard position group:

```json
{ "3": "DL", "7": "DL", "11": "DL", "14": "DL", "22": "LB", "25": "LB", "31": "CB", "38": "CB", "44": "S", "47": "S" }
```

Tokens come from the groups documented in the top-level
[README](../README.md#position-groups): `DL` `LB` `CB` `S`.
