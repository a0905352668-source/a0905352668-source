# Mage-VL Dedicated Host Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将防拍系统的 Mage-VL 复核服务从 `192.168.104.53:8879` 迁移到 `192.168.104.54:8879`，保持对外网页地址和端口不变，并只补跑本次 HTTP 500 失败的复核事件。

**Architecture:** 50-2 继续承载检测、录像、网页和事件服务，只把内部 VLM HTTPS endpoint 改为 `.54:8879`。`.54` 上部署与旧服务哈希一致的 release、模型和锁定依赖，用独立 systemd 单元独占 RTX 3080。切换通过受控 services-only 重载生效，MediaMTX 和 DeepStream 进程身份必须保持不变。

**Tech Stack:** Python 3.12.13, PyTorch 2.11.0+cu128, Transformers 5.12.1, Mage-VL AWQ INT4, HTTPS/TLS, HMAC request signing, systemd, NetworkManager, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-15-mage-vl-migration-design.md`

## Global Constraints

- 不重启、替换或停止 MediaMTX、DeepStream、Docker 推理、摄像头或录像。
- 对外网页 URL、协议、监听地址和端口不得改变；允许网页/事件 services 约 1 分钟受控重载。
- 不停止或修改 `.53` 上的其他 `llama-server`。
- 凭据、共享密钥、TLS 私钥和视频不进 Git，不打印到终端摘要或部署记录。
- 所有生产写入先备份、后原子替换；任一门禁失败立即停止，不用全量重启规避问题。
- 开始生产操作前再读一次 `AGENTS.md`、`README.md` 和本计划，并确认 `git status --short` 没有未识别的修改。

## File Structure

- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/`
  - 从 v179 机械复制，保留已上线的下载功能。
  - Add `live_operator/vlm_requeue.py` 和 `live_operator/tests/test_vlm_requeue.py`。
  - Add `live_operator/services_reload.py` 和 `live_operator/tests/test_services_reload.py`。
  - Add wrappers `scripts/jiankong-vlm-requeue` 和 `scripts/jiankong-reload-services`。
- Create: `01_algorithms/mage_vl_migration/run_mage_vl_service_54.sh`
- Create: `01_algorithms/mage_vl_migration/jiankong-mage-vl-54.service`
- Create: `01_algorithms/mage_vl_migration/tests/test_deployment_contract.py`
- Create after rollout: `01_algorithms/DEPLOYMENT_20260915_MAGE_VL_MIGRATION.md`

---

### Task 1: Freeze the baseline and create the v180 candidate

**Files:**

- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/`
- Reference: `01_algorithms/live_operator/releases/live-operator-20260914-v179-boxed-download/`

- [ ] **Step 1: Record the immutable starting point**

Run:

```bash
cd /tmp/jk-video-download.PE5lkp/repo
git status --short --branch
git rev-parse HEAD
git diff --check
sha256sum 01_algorithms/live_operator/releases/live-operator-20260914-v179-boxed-download/live_operator/{cli.py,vlm_state.py,vlm_review.py}
```

Expected: only this plan/spec documentation is modified before its planning commit; `git diff --check` is silent. Save the commit and three hashes in the execution notes.

- [ ] **Step 2: Mechanically copy v179 to v180**

Run:

```bash
cd /tmp/jk-video-download.PE5lkp/repo
mkdir -p 01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration
rsync -a --exclude='__pycache__/' \
  01_algorithms/live_operator/releases/live-operator-20260914-v179-boxed-download/ \
  01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/
```

Expected: candidate contains the same tracked source files as v179, without bytecode caches.

- [ ] **Step 3: Prove the untouched candidate is equivalent**

Run a sorted SHA256 manifest for both release trees while excluding only the release-local deployment note and `__pycache__`; compare relative path and hash.

Expected: no differences before Tasks 2–3 add their named files.

- [ ] **Step 4: Commit the mechanical candidate**

```bash
git add -f 01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration
git commit -m "chore: stage v180 Mage-VL migration release"
```

### Task 2: Add a fenced, auditable HTTP-500 requeue command

**Files:**

- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/vlm_requeue.py`
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_vlm_requeue.py`
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/scripts/jiankong-vlm-requeue`
- Reference: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/vlm_state.py`

- [ ] **Step 1: Write failing selection tests**

Test the public functions:

```python
plan_http500_requeue(run_dir: Path, config: VLMReviewConfig) -> RequeuePlan
apply_http500_requeue(plan: RequeuePlan, backup_path: Path) -> RequeueResult
```

The fixture must include `pass`, `filter`, `uncertain`, `pending`, retryable and non-retryable `error`, a different error message, and mismatched model/prompt/evidence revisions. Assert that only records satisfying all conditions are selected:

```text
vlm_filter_result == "error"
vlm_filter_error == "VLM review service returned HTTP 500"
vlm_filter_retryable is True
expected model/prompt/evidence revisions exactly match production config
```

Also assert dry-run does not change bytes, mode, mtime, or the lock file.

- [ ] **Step 2: Run the tests and confirm RED**

```bash
cd 01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration
python3 -m pytest -q live_operator/tests/test_vlm_requeue.py
```

Expected: failure because `live_operator.vlm_requeue` does not exist.

- [ ] **Step 3: Implement dry-run planning**

Use `resolve_run_storage(run_dir).metadata_dir / "vlm_filter_states.json"`; reject symlinks and non-regular files, reuse the sidecar size/schema checks, and compute SHA256 from the exact bytes examined. Return sorted event IDs and their SHA256 as an audit fingerprint. Do not read or modify `event_reviews.json`.

CLI contract:

```bash
scripts/jiankong-vlm-requeue --run-dir "$RUN_DIR" --config "$VLM_CONFIG"
```

Default output is one JSON object containing `mode=dry-run`, source SHA256, candidate count, candidate-ID SHA256, and no image or secret material.

- [ ] **Step 4: Write failing apply/locking tests**

Assert that apply:

- requires `--expected-source-sha256` from the dry run;
- takes the same `vlm_filter_states.json.lock` exclusive lock as `VLMReviewStateStore`;
- re-reads and rejects a changed source hash while holding the lock;
- creates an `O_EXCL`, non-symlink, mode `0600` backup containing byte-identical source data;
- atomically removes only selected sidecar entries and fsyncs file and directory;
- is idempotent: a second dry run selects zero;
- aborts when storage authority, run ID, config revisions, error text, or selected set changes.

- [ ] **Step 5: Implement apply and pass the focused suite**

The wrapper requires all of:

```bash
scripts/jiankong-vlm-requeue \
  --run-dir "$RUN_DIR" \
  --config "$VLM_CONFIG" \
  --apply \
  --expected-source-sha256 "$SOURCE_SHA256" \
  --backup-path "$BACKUP_PATH"
```

Expected: JSON reports exactly the removed count and backup SHA256. Removing the compact overlay makes those ready base events new to `_vlm_event_priority`; it does not rewrite the event archive or artificial attempt counters.

Run:

```bash
python3 -m pytest -q live_operator/tests/test_vlm_requeue.py live_operator/tests/test_vlm_state.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -f 01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration
git commit -m "feat: add fenced VLM failure requeue"
```

### Task 3: Add and test the services-only reload path

**Files:**

- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/services_reload.py`
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_services_reload.py`
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/scripts/jiankong-reload-services`
- Reference: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/cli.py`
- Reference: `01_algorithms/DEPLOYMENT_20260914_VIDEO_DOWNLOAD.md`

- [ ] **Step 1: Write failing lifecycle tests**

Define one operation that accepts `--candidate-release`, `--rollback-release`, `--config`, `--candidate-vlm-config`, `--rollback-vlm-config`, `--candidate-ca`, `--rollback-ca`, and `--state`. Mock process hooks and assert:

- it refuses unless state is `running`, all three owned identities are alive, the candidate is a real non-symlink release directory, and the watchdog has `KillMode=process` with empty `ExecStop`;
- it records MediaMTX and DeepStream identities, never calls their stop/start hooks, and only replaces `processes.services`;
- it performs the existing `.stop_events`/fresh `worker_status.json` drain handshake;
- it retains and waits on its own service `Popen`, so a zombie cannot pass identity checks;
- it waits at most 75 seconds for port 8767 to become genuinely reusable;
- on candidate start or HTTP self-check failure it restores the old symlink/config/CA, starts rollback services, verifies readiness, and still leaves MediaMTX/DeepStream identities unchanged;
- an EXIT/finally path always restores the watchdog state.

- [ ] **Step 2: Confirm RED**

```bash
cd 01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration
python3 -m pytest -q live_operator/tests/test_services_reload.py
```

Expected: failure because the module does not exist.

- [ ] **Step 3: Implement the narrow reload command**

Reuse `StateStore`, `ProcessIdentity`, `capture_identity`, `is_same_process`, `_ProcessHooks.block_new_events`, `_ProcessHooks.drain_clips`, and the current service launch environment. Do not expose a generic component name argument: this command is permanently services-only. Persist the replacement identity with the existing state-store lock and atomic save.

The command must print a redacted JSON receipt with before/after identities and return nonzero if any invariant is not proven.

- [ ] **Step 4: Run focused and lifecycle regression tests**

```bash
python3 -m pytest -q \
  live_operator/tests/test_services_reload.py \
  live_operator/tests/test_cli.py \
  live_operator/tests/test_watchdog.py \
  live_operator/tests/test_vlm_state.py \
  live_operator/tests/test_vlm_requeue.py
bash -n scripts/jiankong-reload-services scripts/jiankong-vlm-requeue
```

Expected: all pass, `bash -n` is silent.

- [ ] **Step 5: Commit**

```bash
git add -f 01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration
git commit -m "feat: add guarded services-only reload"
```

### Task 4: Add the dedicated `.54` service deployment contract

**Files:**

- Create: `01_algorithms/mage_vl_migration/run_mage_vl_service_54.sh`
- Create: `01_algorithms/mage_vl_migration/jiankong-mage-vl-54.service`
- Create: `01_algorithms/mage_vl_migration/tests/test_deployment_contract.py`

- [ ] **Step 1: Write failing contract tests**

Parse the runner and systemd unit as text. Assert exact endpoint `192.168.104.54:8879`, GPU `0`, weight memory `3800MiB`, CPU memory `24GiB`, max request bytes `67108864`, model version `mage-vl-awq-v20260809-1f7f5266fa4e-phone-use-prompt-v2`, absolute `/home/zty/YL/JianKong` roots, TLS/secret/cache/GPU-lock paths, `User=zty`, `Restart=on-failure`, and no embedded credential material. Assert the runner rejects symlinked or non-owned private files and verifies the current release stays under `releases/`.

- [ ] **Step 2: Confirm RED, then implement the runner and unit**

```bash
python3 -m pytest -q 01_algorithms/mage_vl_migration/tests/test_deployment_contract.py
bash -n 01_algorithms/mage_vl_migration/run_mage_vl_service_54.sh
```

Runner environment must set `CUDA_VISIBLE_DEVICES=0`, `PYTHONPATH` to the current standalone Mage release, and execute the dedicated venv Python with `-m live_operator.mage_vl_service`. Runtime files remain outside Git.

- [ ] **Step 3: Commit**

```bash
git add -f 01_algorithms/mage_vl_migration
git commit -m "ops: define dedicated Mage-VL host service"
```

### Task 5: Stage and verify `.54` without touching production

**Hosts:** source `build` (`192.168.104.53` through 50-2), target `zty@192.168.104.54` through `build`.

- [ ] **Step 1: Recheck both hosts and the source process identity**

Run read-only checks for hostname, OS/glibc, driver, GPU inventory/processes, free RAM/disk, source current-release realpath, model realpath, service PID/cmdline/start token, Python version, `pip freeze --all`, and SHA256 manifests. Abort if `.54` GPU is not idle or source paths/revisions differ from the approved design.

- [ ] **Step 2: Create a reproducible dependency bundle**

On `.53`, install `conda-pack==0.8.1` into a newly created temporary directory only, package `/home/build/miniconda3/envs/bigmodel`, capture the source venv's `pip freeze --local --all`, and download the overlay wheels into the staging directory. Do not install into or modify the running source environment.

Expected artifacts under a mode-0700 timestamped staging directory: conda archive, overlay requirements, wheelhouse, release manifest, model manifest, and a top-level SHA256SUMS. If every overlay requirement cannot be satisfied from the wheelhouse, abort rather than substituting a newer dependency.

- [ ] **Step 3: Transfer source-to-target and verify hashes**

Use `rsync -aH --no-owner --no-group` over the existing `.53` to `.54` LAN path for the standalone current release, AWQ model, conda archive, and wheelhouse; verify the resulting files are owned by `zty`. Install to:

```text
/home/zty/YL/JianKong/01_algorithms/mage_vl_service/releases/mage-vl-service-20260812-v25-temporal-early8
/home/zty/YL/JianKong/07_models/vlm_models/versions/v20260809_01_mage_vl_awq_int4
/home/zty/YL/JianKong/08_envs/bigmodel-20260915
/home/zty/YL/JianKong/08_envs/mage-vl-20260915
```

Unpack the base environment, run `conda-unpack`, create the overlay venv from its Python 3.12.13, install only from the transferred wheelhouse with `--no-index`, and compare effective `pip freeze --all` against the source. Run an absolute old-prefix scan; any `/home/build/` reference in active Python metadata is a blocker.

- [ ] **Step 4: Generate target TLS and install private runtime files**

Generate a new self-signed TLS certificate on `.54` with CN `jiankong-mage-vl-54` and SAN `IP:192.168.104.54`. Keep key, shared secret, GPU lock, runtime directory, and cache owned by `zty`; modes: directories `0700`, private files `0600`, certificate `0644`. Transfer the existing shared secret directly host-to-host without displaying it.

- [ ] **Step 5: Install and start the systemd unit**

Install the reviewed runner and unit, run `systemd-analyze verify`, then `enable --now jiankong-mage-vl-54.service`. Verify:

```text
systemctl is-active == active
ss shows only 192.168.104.54:8879
the unit's owned PID is the only compute process on GPU 0
health identity returns the exact model/prompt/evidence revisions
```

Do not stop the old Mage service yet.

### Task 6: Add the exact route and run signed pre-cutover probes

**Host:** 50-2 only.

- [ ] **Step 1: Capture production invariants and rollback material**

Derive the active run from `/media/boshi/Data/JianKong/02_configs/runtime/live_operator_state.json`. Record config/current-release hashes, public URL/listener, watchdog unit properties, 8/8 source status, aggregate FPS, source errors, and full `ProcessIdentity` fields for MediaMTX, services, and DeepStream. Copy the VLM config and current CA to mode-0600 timestamped rollback files on the same filesystem.

- [ ] **Step 2: Add only the `.54/32` route**

Verify `windows-jump` still owns `.53/32 via 192.168.0.1`. Persist `192.168.104.54/32 192.168.0.1` in that same profile, then add the identical runtime route on `enp3s0f0`. Verify:

```bash
ip route get 192.168.104.54
timeout 3 bash -c 'exec 3<>/dev/tcp/192.168.104.54/8879'
```

Expected route includes `via 192.168.0.1 dev enp3s0f0`; TCP succeeds. Confirm no `/24` route was added. Rollback is removal of this exact `/32` only.

- [ ] **Step 3: Install the public certificate without changing production config**

Copy `.54`'s public certificate to `/media/boshi/Data/JianKong/02_configs/runtime/vlm_review/tls-ca-192.168.104.54.crt` through a temporary file, validate SAN/fingerprint, set mode `0644`, and atomically publish. Do not copy the private key.

- [ ] **Step 4: Run authenticated health and real-evidence probes**

Create a mode-0600 temporary probe config under `/run/user/$(id -u)` by copying production VLM config and changing only endpoint and CA path. Load it with `VLMReviewConfig.load`, run `VLMReviewClient.check_health()`, then select one completed recorded event from the active run and call `VLMReviewClient.review()` with its immutable clip and overlay using a migration-only request ID.

Expected: HTTP 200, valid response signature, event/request IDs match, and result/model/prompt revisions validate. Compare sidecar and event archive hashes before/after; they must be unchanged. Delete the temporary probe config.

### Task 7: Cut over the internal endpoint with services-only reload

**Files installed on 50-2:** v180 candidate release and its two operational wrappers.

- [ ] **Step 1: Install and verify v180**

Transfer the candidate into `/media/boshi/Data/00_active_projects/JianKong/01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration`, compare every tracked Git blob/hash, and run the focused test suites with production's Python 3.13. Do not switch `current` yet.

- [ ] **Step 2: Prepare the exact config delta**

Under a mode-0600 temporary file, parse the current JSON and change only:

```json
{
  "endpoint": "https://192.168.104.54:8879/v1/review",
  "tls_ca_file": "/media/boshi/Data/JianKong/02_configs/runtime/vlm_review/tls-ca-192.168.104.54.crt"
}
```

Assert every other key/value is byte-semantically identical, then validate the temporary file with `VLMReviewConfig.load` and its signed health check.

- [ ] **Step 3: Execute the guarded cutover**

Stop the watchdog through systemd, atomically switch `current` to v180 and publish the prepared VLM config, then invoke `jiankong-reload-services` with v179 and the VLM config/CA rollback paths. The tool must either verify new HTTP readiness or restore v179 plus old config/CA and old services. Its finally path restores the watchdog.

Expected outage: only the dashboard/event service, at most about one minute. Never call `jiankong-restart`.

- [ ] **Step 4: Prove invariants immediately**

Compare recorded process identities, not only PIDs. MediaMTX and DeepStream must be byte-for-byte unchanged; only services must have a new identity. Verify watchdog active/healthy, 8/8 cameras, aggregate FPS near 80, zero source errors, dashboard health, events API, video playback, and download endpoint. Verify the externally used page URL and listener tuple equal the pre-cutover values.

If any invariant fails, run the tool's config/release rollback path and stop; do not attempt a full restart.

### Task 8: Observe, requeue the failed batch, and retire old Mage-VL

- [ ] **Step 1: Observe the new endpoint before backfill**

Require both: five consecutive newly completed VLM reviews and ten continuous minutes with no new `VLM review service returned HTTP 500`. During the window, sample `.54` GPU memory/processes, system memory, service journal, 50-2 queue status, source FPS/errors, and unchanged MediaMTX/DeepStream identities.

- [ ] **Step 2: Dry-run the exact backfill**

On 50-2:

```bash
STATE=/media/boshi/Data/JianKong/02_configs/runtime/live_operator_state.json
VLM_CONFIG=/media/boshi/Data/JianKong/02_configs/runtime/vlm_review.json
RUN_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "$STATE")"
CURRENT=/media/boshi/Data/00_active_projects/JianKong/01_algorithms/live_operator/current
"$CURRENT/scripts/jiankong-vlm-requeue" --run-dir "$RUN_DIR" --config "$VLM_CONFIG"
```

Cross-check the candidate count against an independent read-only JSON count. Review the candidate-ID SHA256 and source SHA256 before apply.

- [ ] **Step 3: Apply once under the authoritative lock**

Set `SOURCE_SHA256` to the dry-run field and derive a timestamped backup path in the same metadata directory. Run the apply CLI with the exact dry-run hash. Expected: backup hash equals source hash; removed count equals candidate count; all unrelated state entries remain byte-semantically identical; backup mode is `0600`.

- [ ] **Step 4: Monitor completion and reconcile counts**

Let the existing single-worker queue process the events. Report original candidate count, new pass/filter/uncertain/error/pending counts, and any event still absent. Verify no manual review record changed and a second dry run selects zero. Do not repeatedly reset newly failed records.

- [ ] **Step 5: Stop only old Mage-VL after stability is proven**

On `.53`, resolve supervisor and Mage service PIDs from their owned runtime files, verify UID/start token/cmdline/current release, send TERM only to those two owned processes, and wait boundedly for exit. Verify the unrelated `llama-server` PID/cmdline/start token is unchanged. Preserve old release, model, cache, configuration, TLS material, and logs.

### Task 9: Final verification, receipt, and Git synchronization

**Files:**

- Create: `01_algorithms/DEPLOYMENT_20260915_MAGE_VL_MIGRATION.md`

- [ ] **Step 1: Run final verification**

Run all v180 Python tests plus `bash -n` and deployment contract tests. Repeat production health, page/event/video/download checks, 8/8/FPS/source-error checks, process-identity comparison, `.54` exclusive GPU ownership, signed VLM health, and backfill reconciliation. Run `git diff --check`.

- [ ] **Step 2: Write the evidence-based deployment receipt**

Record timestamps, Git commit, installed release/model/environment hashes, route, public listener before/after, old/new endpoint without credentials, process identities, service observation results, backfill counts, rollback artifacts and SHA256, and the fact that no detection/recording process restarted. Do not record secrets, private keys, passwords, event imagery, or unredacted journals.

- [ ] **Step 3: Commit**

```bash
git add -f \
  01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration \
  01_algorithms/mage_vl_migration \
  01_algorithms/DEPLOYMENT_20260915_MAGE_VL_MIGRATION.md
git diff --cached --check
git commit -m "ops: migrate Mage-VL review service to dedicated host"
```

- [ ] **Step 4: Synchronize Git without overstating success**

Push `codex/boxed-video-download-20260914` to GitHub. If GitHub remains unreachable, create a Git bundle, import it into `/media/boshi/Data/00_active_projects/JianKong`, update only the same branch ref there, verify both commit IDs match, and report GitHub push as pending rather than successful.

- [ ] **Step 5: Apply completion verification discipline**

Before claiming completion, invoke `superpowers:verification-before-completion` and cite fresh command output for every success condition above.
