# V5 Memo decommissioning

Status: **deprecated and not publishing** as of 2026-07-19.

V5 was decommissioned after repeated failure to sustain dependable daily
publication. Its production timers, publisher services, backup timer, and
dedicated search service are disabled. No new V5 submissions should be made.

The repository, historical receipts, state, and caches are intentionally
retained for audit and a reversible rollback. Historical public artifacts are
not removed.

This decommissioning is scoped to V5 only. It does not deprecate or change the
Researka platform, shared database/search services, or other research-agent
lanes.

Reactivation requires all of the following:

1. An explicit owner decision to reactivate V5.
2. A review of the retained runtime state and pending submissions.
3. Clean tests, lint, and strict typing on the chosen reactivation commit.
4. A fresh unattended end-to-end proof producing a listed public artifact and
   minted DOI before any daily-cadence claim.
