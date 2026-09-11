# Moving-person report gate — 2026-09-09

Approved scope: suppress alarm AND review while a tracked person clearly moves,
including walking while aiming a phone. Once stopped, accumulate original gate
evidence afresh. Do not change phone/model thresholds, cooldown, static-object
policy, cameras, FPS, frontend, media storage, or Mage-VL prompt.

Movement uses raw body-box center plus two reliable shoulder midpoint positions,
not wrists or phones. Per-track window0.7–1.05s, ≥5valid samples. Body net
translation≥0.12person-height and shoulder translation≥0.10height; both agree
in direction and body trajectory efficiency≥0.75. Stop hysteresis: ≥0.8s
compact body-position span<0.10height, shoulders displacement<0.12height.
Short missing shoulders bridge up to0.5s; stale/invalid identities expire.
One-step jumps>0.75height reset history. Default 8/10FPS gate cadence unchanged.

Limitations: not a learned walking classifier. Very slow/subtle motion or long
shoulder occlusion may not be confidently detected and is not blindly suppressed.
Reliable moving classification needs a short observation interval. Existing events
are not retroactively removed. Short gaps/ID switches can affect tracking.

Tests: standalone motion/gate tests and actual pipeline --self-test-rules
(both legacy/spatial branches). Integration RED exit31 before wiring. Independent
review found periodic shoulder loss and bounded sway release issues; both gained
RED/GREEN regressions. Existing moving-handheld replay fixture now expects the
new motion suppression, while preserving original static-history assertions.
One existing CTest contract suite requires absent tools/validate_static_phone_regression.py;
this environment limitation is reported, not masked as a fully passing suite.

Deployment changes only the manifest-selected inference binary/source. Keep v178
hot-aware Dashboard untouched; rollback only the inference manifest, never v177.
