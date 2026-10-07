# Operations

Scheduled jobs for a long-running install. Everything here runs as your user; nothing needs root.

Automate catalog updates and canary checks with `systemd --user` or `crontab`.

## Security & Environment

Store API keys outside the repository (e.g. `~/.config/whichapiapi/env`). Never commit keys. On the owner's server
every key also lives in the shared folder `~/.config/secrets/<service>.env` (index in its `README.md`); systemd
units can list several `EnvironmentFile=` lines. `offers pull-benchmarks` adds Artificial Analysis only when
`ARTIFICIAL_ANALYSIS_API_KEY` is set.
Always set `WHICHAPIAPI_BUDGET_TOTAL` as a global real-money ceiling for commands that can spend.

Example `~/.config/whichapiapi/env`:
```bash
WHICHAPIAPI_BUDGET_TOTAL=1.5
RESELLER_API_KEY=sk-...
```

---

## 1. Daily Catalog Refresh

Refreshes models.dev, OpenRouter, an optional new-api reseller channel, Arena Elo quality scores, and discount conditions. Pulls are free (no model calls).

### Systemd User Units

`~/.config/systemd/user/whichapiapi-catalog.service`:
```ini
[Unit]
Description=WhichAPIAPI daily catalog refresh

[Service]
Type=oneshot
WorkingDirectory=%h/projects/which_apiapi
EnvironmentFile=%h/.config/whichapiapi/env
ExecStart=%h/.local/bin/uv run --project %h/projects/which_apiapi whichapiapi offers pull all --url https://api.reseller.example --channel reseller --key-env RESELLER_API_KEY
ExecStart=%h/.local/bin/uv run --project %h/projects/which_apiapi whichapiapi offers pull-quality
ExecStart=%h/.local/bin/uv run --project %h/projects/which_apiapi whichapiapi offers pull-conditions
```

`~/.config/systemd/user/whichapiapi-catalog.timer`:
```ini
[Unit]
Description=Daily run for WhichAPIAPI catalog refresh

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable via: `systemctl --user daemon-reload && systemctl --user enable --now whichapiapi-catalog.timer`.

---

## 2. Daily Canary Run

Detects silent provider-side model substitutions or latency regressions on `examples/canary/suite.yaml`.

To bypass the confirmation prompt (`typer.confirm`), pass `-y` / `--yes`.

### Systemd User Units

`~/.config/systemd/user/whichapiapi-canary.service`:
```ini
[Unit]
Description=WhichAPIAPI daily canary check

[Service]
Type=oneshot
WorkingDirectory=%h/projects/which_apiapi
EnvironmentFile=%h/.config/whichapiapi/env
ExecStart=%h/.local/bin/uv run --project %h/projects/which_apiapi whichapiapi eval canary examples/canary/suite.yaml -y --budget 0.05
```

`~/.config/systemd/user/whichapiapi-canary.timer`:
```ini
[Unit]
Description=Daily run for WhichAPIAPI canary suite

[Timer]
OnCalendar=*-*-* 04:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable via: `systemctl --user daemon-reload && systemctl --user enable --now whichapiapi-canary.timer`.

---

## 3. Leaderboard auto-refresh of your own suites

`eval refresh <suite.yaml>` compares the suite's providers with today's best models for its task
(`x-whichapiapi.task`: general, coding, math, agents, russian, writing, long_context, vision) that your keys can call.
Dry run by default; `--write` appends the new ones (with the current provider's `config`), `--run` re-runs the suite
under its budget — cached results keep old providers free, so only new models (and their judge calls) cost money.
Exit code **3** = the recommended primary changed (also logged to the activity log as `refresh`).

Weekly, after the catalog refresh:

```ini
# ~/.config/systemd/user/whichapiapi-suite-refresh.service
[Service]
Type=oneshot
WorkingDirectory=%h/my-suites
EnvironmentFile=%h/.config/secrets/reseller.env
Environment=WHICHAPIAPI_BUDGET_TOTAL=1.5
ExecStart=%h/.local/bin/uv run --project %h/projects/which_apiapi whichapiapi eval refresh my-suite/suite.yaml --write --run -y --budget 0.10
```

With a `.timer` (`OnCalendar=Mon 05:00`) and `OnFailure=` pointing at a notifier, exit 3 becomes an alert.
**Enabled on the owner's server (owner's yes, 2026-09-29):** `whichapiapi-suite-refresh.timer`, Mondays 05:30 UTC,
working copy under `~/.local/share/whichapiapi/suites/` (so `--write` never touches the repo), `--budget 0.15`,
`WHICHAPIAPI_BUDGET_TOTAL=1.5`, a reseller `min_balance: 0.3`. A new winner shows as a failed unit
(`systemctl --user --failed`) and as a `refresh` event in `whichapiapi activity`; the runs dashboard is rebuilt
after every run (`ExecStopPost`). First run ≈ $0.11 (mimo-v2.6-pro + uncached verdicts); weeks with no new model ≈ $0.

---

## Alternative: Crontab

Edit with `crontab -e`:

```cron
REPO=/home/user/projects/which_apiapi
BIN=/home/user/.local/bin/uv
ENV=/home/user/.config/whichapiapi/env

0 3 * * * set -a; . $ENV; set +a; $BIN run --project $REPO whichapiapi offers pull all --url https://api.reseller.example --channel reseller --key-env RESELLER_API_KEY && $BIN run --project $REPO whichapiapi offers pull-quality && $BIN run --project $REPO whichapiapi offers pull-conditions
0 4 * * * set -a; . $ENV; set +a; $BIN run --project $REPO whichapiapi eval canary $REPO/examples/canary/suite.yaml -y --budget 0.05
```

---

## Viewing Logs & Canary Alerts

Check service execution logs:
```bash
journalctl --user -u whichapiapi-catalog.service
journalctl --user -u whichapiapi-canary.service
```

### Canary Alerts
`cli.py` has no separate query command for historical canary alerts. Alerts are output during execution:
- `eval canary` exits with code **2** when it raises alerts, so a failed unit (`systemctl --user --failed`),
  an `OnFailure=` hook or cron's `MAILTO` can notify you.
- View past alerts in systemd output:
  ```bash
  journalctl --user -u whichapiapi-canary.service --since -7d -g ALERTS
  ```
- Reset baseline after inspecting an acknowledged alert:
  ```bash
  uv run --project <repo> whichapiapi eval canary examples/canary/suite.yaml -y --reset-baseline
  ```
- Inspect spend ledger:
  ```bash
  uv run --project <repo> whichapiapi spend
  ```

---

## Gaps

- No CLI command exists to query historical canary alerts or list stored baselines from the database (alerts only stream to stdout during `eval canary`).
- `offers pull all` only accepts one `--url`/`--channel`/`--key-env` tuple; pulling multiple new-api resellers requires individual `offers pull newapi` invocations.
- No non-zero exit code is produced when canary alerts are raised (returns exit code 0 unless an unhandled error/budget failure occurs).

## Installed on the owner's server

`~/.config/systemd/user/whichapiapi-refresh.{service,timer}` — daily 04:30 UTC (±15 min): `offers pull all` (+ the
a reseller channel), `offers pull-benchmarks` (Artificial Analysis when its key file exists), `offers pull-conditions`,
`models export` and `models page` into `~/.local/share/whichapiapi/`. Keys come from `~/.config/secrets/*.env`.
Logs: `journalctl --user -u whichapiapi-refresh`.
