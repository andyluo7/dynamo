# Phase 0 Inventory — AAC1 MI355X PoC environment

Date: 2026-05-11
Allocated node: `smci355-ccs-aus-g12-22` (job 56, holder, 8h)
Login: `aac1.amd.com`, user `anluo`

## Node hardware

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-111-generic |
| CPU | 2x AMD EPYC 9575F 64-core (128 cores total) |
| GPU | 8x AMD Instinct MI355X (gfx950, sramecc+, xnack-, device 0x75a3) |
| RAM | 1500 GB (per `scontrol show node` on g12-18 — same node class) |
| Local SSD | `/dev/nvme0n1p2` 7.0 TB at `/`, 6.6 TB free |
| NIC | **AMD Pensando RoCE HCA** (Pensando Device 1012) — 9 `ionic_*` interfaces (`ionic_0..ionic_8`); firmware `1.117.5-a-11` |
| Shared FS | NFS `10.194.30.70:/AAC_xai_home` mounted at `/shared/amdgpu/home`, 9.5 TB total / 6.8 TB free |

## Toolchain

| Tool | State |
|---|---|
| ROCm | Not on host PATH; modules at `/shared/apps/modules/ubuntu/modulefiles`. Available: `rocm/7.2.1`, `rocm/7.2.2`, plus `rocm-7.2.x/ucx/1.19.0` and `+ompi/5.0.9` variants. ROCm tree at `/shared/apps/ubuntu/opt/rocm-7.2.2/`. Use **`bash -lc 'module load rocm/7.2.2'`** to activate (login shell required for `module`). |
| Container | **Use podman** (`/usr/bin/podman`, v4.7.1). Docker is installed (v29.2.1) but user is NOT in `docker` group (`groups` = `anluo video render slurmusers QLE`). |
| ibv tools | Not installed on host (no `ibv_devinfo`, `ibv_devices`). Probe via `/sys/class/infiniband/ionic_*/` instead, or run inside a container with `libibverbs-dev`/`ibverbs-utils`. |
| HF tooling | `huggingface_hub` 1.14.0 importable on login node. `huggingface-cli` not on PATH. **No HF token configured** (no env var, no `~/.cache/huggingface/token`); unauthenticated downloads are working but rate-limited. |

## ionic NIC configuration (CRITICAL for Phase 3/4)

| Item | Status |
|---|---|
| Number of ionic devices | 9 (`ionic_0` through `ionic_8`) — fork's runbook expects 8, so verify whether `ionic_8` is admin/management or counted in pool |
| Driver | `AMD Pensando RoCE HCA`, fw `1.117.5-a-11` |
| Port state | At least `ionic_0` port 1 is `ACTIVE` |
| **IPv4 GIDs** | **Already configured** by cluster admins. `ionic_0` `GID[1] = 0000:...:ffff:acc1:0104` → `172.193.1.4` (matches `enp121s0`). This means `setup_ionic_network.sh` from the fork is **not needed**. |
| Container ABI fix | Still TBD — need to verify host vs container `libionic` version match when we pull a Dynamo image. Fork uses `docker cp $(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.1.* | head -1) <container>:/usr/lib/x86_64-linux-gnu/libionic.so.1` |

**Implication for plan:** Phase 3/4 patch surface is **smaller** than initially estimated because IPv4 GID setup is already done. The remaining ionic-tax items from the fork (libionic ABI fix, MoRI CQE patch, MoRI prewarm, SearchBySubnet) are still required if we want MoRI/Mooncake transports. NIXL/UCX with `UCX_TLS=rc_v,tcp` is the simplest first attempt.

## Routable network interfaces (login candidate for cross-node etcd/NATS)

```
enp196s0   10.194.30.27/24  fe80::690:81ff:fe64:3a58  (mgmt subnet, login network)
enp121s0   172.193.1.4/31   (ionic_0 IPv4 — RoCE)
enp9s0     172.1.1.4/31     (ionic_? IPv4)
enp105s0   172.161.1.4/31
enp25s0    172.225.1.4/31
enp249s0   172.65.1.4/31
enp137s0   172.129.1.4/31
... (8 ionic IPv4 interfaces)
```

Per fork runbook: use `ip route get 1.1.1.1 | awk '/src/ {print $7}'` to get the management IP (NOT `hostname -I`, which returns ionic IPs first).

## Cached models on `/shared/amdgpu/home/anluo`

| Model | Path | Size |
|---|---|---|
| `MiniMaxAI/MiniMax-M2.5` | `inferencex-agentic-test/hf-cache/hub/models--MiniMaxAI--MiniMax-M2.5` | 216 GB |
| `DeepSeek-R1-0528-MXFP4-Preview` | `models/DeepSeek-R1-0528-MXFP4-Preview` | 377 GB |
| `amd/Kimi-K2.5-MXFP4` | `inferencex-agentic-test/hf-cache/models--amd--Kimi-K2.5-MXFP4` | 523 GB |

DSR1-FP8 (`deepseek-ai/DeepSeek-R1-0528`) download started 2026-05-11 13:25 to `/shared/amdgpu/home/anluo/.cache/huggingface/hub/`. Background PID on login node was 535260. Log: `/shared/amdgpu/home/anluo/dsr1_download.log`.

## Open items for Phase 0 close-out

- [ ] Verify download completes (check `dsr1_download.log` and the snapshot dir)
- [ ] Pull a Dynamo container image to validate podman + GPU + ROCm stack end-to-end (likely `rocm/dynamo-vllm:latest` or build from `Dockerfile.rocm-sglang` with podman)
- [ ] Inside container, run `ibv_devinfo` to confirm ionic visibility and check libionic ABI compatibility
- [ ] Raw-backend baseline: serve `MiniMax-M2.5` with stock vLLM-ROCm container (no Dynamo) and capture P50 / throughput at c=1, c=4
- [ ] Raw-backend baseline: serve `DeepSeek-R1-0528` FP8 with stock SGLang-ROCm container (once download completes)
