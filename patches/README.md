# Patches

Local patches applied to git submodules. We keep these as patch files (rather than maintaining a fork) because the fixes are small and the submodule is rarely updated.

## How to apply

After cloning, run `git submodule update --init --recursive` and then:

```
./patches/apply.sh
```

The script is idempotent — re-running it is a no-op if the patches are already applied.

## Patches

| File | Submodule | Purpose |
|---|---|---|
| `0001-greenportscan-use-self-agent.patch` | `CybORG_plus_plus` | `GreenPortScan` looked up the source session in `state.sessions['Red']` (hardcoded), so it crashed for any non-zero green session. Fixed to use `state.sessions[self.agent]` like `GreenConnection` does. Without this fix the stock `GreenAgent` only works for `session=0` (red's foothold), and our `BehavioralGreenAgent` can't dispatch traffic from User1–4 / Op_Host0–2. |
