# Mage-VL dedicated-host migration deployment receipt

Deployment window: 2026-09-15 through 2026-09-16 (Asia/Shanghai)

Reviewed branch: `codex/boxed-video-download-20260914`

Public service: `http://192.168.50.2:8767/` (unchanged)

## Result

The anti-screen-capture system's Mage-VL review backend moved from the shared
host at `https://192.168.104.53:8879/v1/review` to the dedicated RTX 3080 host
at `https://192.168.104.54:8879/v1/review`. The public web address and port did
not change. The guarded cutover replaced only the `services` process; it did
not intentionally restart MediaMTX, DeepStream, cameras, recording, or GPU
detection. The old `.53` Mage service and its persistent cron launch entries
were subsequently retired while its release, model, logs, TLS material, and
rollback artifacts were retained.

The live release is:

```text
live-operator-20260915-v180-mage-vl-migration
```

The live VLM configuration uses the canonical `.54` endpoint and its dedicated
public CA. A persistent `192.168.104.54/32` route through `192.168.0.1` was
added to the existing 50-2 `windows-jump` NetworkManager profile; the existing
`.53/32` route was preserved and no broader `/24` route was added.

## Reviewed identities and artifacts

- Mage release: `mage-vl-service-20260812-v25-temporal-early8`
- Model: `v20260809_01_mage_vl_awq_int4`
- Python: `3.12.13`
- Model revision:
  `mage-vl-awq-v20260809-1f7f5266fa4e-phone-use-prompt-v2`
- Prompt revision:
  `600887293a7b364de10ee857979dd0713903914cc22212a66ee6c55169b7b993`
- Evidence revision:
  `person-roi20-span5s-pre4-native-focus-temporal-early8-jpeg92-v20`
- Packed base environment SHA-256:
  `98ac8c7832e6a40ef9f6a4483a89d914794fa3b2dfaa5d82a9acecf23599a130`
- Release manifest SHA-256:
  `fad5d2fea956ef4ea663e44a3bb04c523d5d9703ae5d9808a205b9a2e493f79d`
- Model manifest SHA-256:
  `b74fbc04dc4101591a797b08160ebff1e0e0b1f583cebddbffa3c2bccbb3df33`
- Exact `ms-swift` source manifest SHA-256:
  `dbabf8f622e21b411cf782f93f421f2f057aa754a27faaf903cdee31a48ca86c`
- Target normalized effective-freeze SHA-256:
  `a3bb1525d8f7c523b771cb3dbf79d3670a0f14f4b2b68f8e547059e19b9abb96`
- Reviewed runner SHA-256:
  `b144f8b3252b4f6efb1c7ff4cd65b2baa649780f492de45ddbc6f4339ebfef88`
- Reviewed systemd unit SHA-256:
  `76e824f32d035695d088d305b3ebe49f8daa84d67fe131564e3deda58f8caec3`
- Installed 50-2 `.54` CA SHA-256:
  `67d9dd65ec1d0b16effabfce0a6403a9963ebf7e779ef4fe3ab7c5c6d6a15876`

Release, model, environment, editable dependency, TLS, runtime permissions,
systemd parent authority, target listener, sole GPU ownership, and signed
model/prompt/evidence health were independently checked before cutover. No
secret, private key, password, event identifier, image, or request payload is
recorded in this receipt.

## Cutover and runtime invariants

The successful guarded transaction reported `ok=true`, `rolled_back=false`,
and `watchdog_restored=true`. At that boundary, MediaMTX and DeepStream retained
their exact PID/start-token/PGID identities; only `services` changed and took
ownership of `0.0.0.0:8767`. The selector and VLM configuration were published
atomically to v180 and `.54` respectively.

50-2 rebooted at `2026-09-16 07:00:23 +08:00`, after that receipt. Therefore
the earlier PIDs are not claimed to have survived the reboot. A fresh
post-reboot baseline was established instead:

```text
MediaMTX:   17969 / 33873 / 17969
DeepStream: 18010 / 34158 / 18010
services:   18327 / 34695 / 18327
watchdog:   PID 4020, active/running, NRestarts=0
```

Fresh Task 9 read-only checks found those three post-reboot identities still
alive, v180 still selected, the live configuration still pointing to `.54`,
signed health valid, watchdog healthy with zero restarts, and the target unit
active/running with PID `1504864`, zero restarts, one `.54:8879` listener, and
the same PID as the sole GPU compute process. Public checks returned 8/8
cameras, about 79.9 aggregate FPS, zero source errors and alerts, HTTP 200 for
the dashboard/events API, and HTTP 206 for both clip and already-generated
boxed-video Range requests.

## Fenced historical requeue and reconciliation

The v180 dry-run and an independent parser selected exactly 626 records from
run `live_20260915_070041`. The one apply used both the source and intent
fences, created a same-directory `boshi`-owned mode-0600 byte-identical backup,
removed only the selected overlays, and preserved the other 235 overlays and
the event/manual-review archives. The backup source SHA-256 is
`49c48c6535c8d2e63b22095865b7fff23af1e5ea2eb4be0063143c2947108ed8`;
the candidate-set SHA-256 is
`a8912714942fd556247a03a982cc0be80582b08d8e529d44183163a0b9442779`;
the canonical intent SHA-256 is
`295081e84cfd03a3b760b357647e7840069bdf3bcde70fcc692ce703d35a1252`.
Repeated dry-runs selected zero after apply.

The original 626 remain under the existing single worker's background
reconciliation. The fresh Task 9 snapshot was:

```text
pass=56 filter=27 uncertain=0 error=0 pending=1 absent=542
```

This is progress, not a claim of full reconciliation. Completion requires all
626 records to become terminal; the remaining counts will continue to change.

## `.53` retirement and rollback material

The old supervisor and Mage child were signalled only after exact identity,
ownership, parentage, listener, and PID-file checks. A cron relaunch was caught,
the two exact old-Mage launch lines were removed, and the relaunched exact PIDs
were then stopped. Fresh Task 9 checks found zero old Mage processes, no
supervisor PID file, no `.53:8879` listener, and no remaining Mage cron entry.
The unrelated llama process retained PID/start token `1672922/345526166` and
full-command SHA-256
`7968e7e1be73a1cfa45e35ac88cf71474a07a58b2d3c2df6688ded66261abd65`.

Rollback material retained includes:

- the unchanged v179 release and Task 7 rollback VLM configuration;
- the pre-route live VLM configuration and old public-CA backups on 50-2;
- the exact `.53` crontab backup at mode 0600, whose source SHA-256 is
  `acf23ebb45f360fdea6deb4aa101d4c4dd38724d8ca616426630f4d996430ec8`;
- the old `.53` release, model, cache, configuration, TLS material, and logs;
- the exact `.54/32` route rollback documented in the Task 6 execution report.

Restoring `.53` is not a single-step rollback: it requires restoring its cron
launcher, proving the old service identity and signed health, switching the
guarded live configuration/selector back through the reviewed transaction,
and only then considering removal of the `.54/32` route.

## Verification and known test debt

Task 9 deliberately honored the request for reduced testing. One effective
fresh focused run produced:

```text
v180 cutover suites:          167 passed, 4 skipped
Mage deployment contracts:   4 passed, 7 skipped
three Bash wrapper checks:    PASS
git diff --check:             recorded at final staging
```

The skips are platform-specific Linux checks in the local macOS environment;
the deployment contract's Linux behaviors had already passed on 50-2 during
Task 4. An initial Task 9 invocation from the repository root was discarded
because it could not import the release package; it collected no tests and was
rerun from the release root shown above.

The earlier broad diagnostic remains explicit debt: **385 tests passed and 37
inherited or non-isolated tests failed**. Those failures include stale
five-second clip assertions against the 17-second contract, tests that consume
production absolute configuration, and a MediaMTX test referencing an installer
not shipped in the release candidate. They were not rerun or represented as
passing during this reduced final gate.

## Reviewed commits and synchronization

The migration is represented by commits `5c39409` through `9eddd43`, including
the v180 snapshot, fenced requeue, guarded services-only reload, dedicated
`.54` service contract, canonical path handling, and systemd-249 watchdog
compatibility. This receipt is committed with the final operations commit.

At receipt creation, the local branch contained the complete reviewed history;
GitHub synchronization remained pending from earlier network failures. Task 9
performs one final push attempt after creating the receipt commit. If GitHub is
still unavailable, the authoritative final result (verified bundle hash,
protected 50-2 backup/import location, updated branch ref or exact safe pending
action) is recorded in the ignored Task 9 execution report because the receipt
itself is part of the commit being synchronized. The dirty 50-2 `main` checkout
must never be reset, checked out, overwritten, or made inconsistent to advance
this separate branch.
