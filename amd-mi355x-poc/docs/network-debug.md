# Network debug — ionic NICs on AAC1

What we learned about the cluster fabric while tracking down the MoRI
c≥4 wall. Short answer: **AAC1 ionic runs lossy RoCEv2 with ECN/DCQCN
congestion control, not lossless PFC.** That choice is correct for
Mooncake and RIXL; wrong for MoRI.

## Direct evidence

After our Phase 6b/c MoRI runs (job 82) split cleanly between c=1
success and c≥4 failure, we read the ionic NIC sysfs hw_counters on
`smci355-ccs-aus-g12-06` directly:

```
rx_rdma_ecn_pkts:       924,933,861   ← 925M ECN-mark events
rx_rdma_cnp_pkts:        17,954,708   ← DCQCN Congestion Notification Packets
rdma_puec_cc_cwnd_dec:  798,930,482   ← 798M cwnd shrinks (vs 5.7M grows)
rdma_retx_rto:               16,218   ← retransmission timeouts
resp_rx_dup_request:     15,077,283   ← duplicate RDMA requests
```

Presence of `rx_rdma_ecn_pkts` and `rx_rdma_cnp_pkts` plus absence of
any PFC pause counters confirms **lossy RoCEv2 + ECN/DCQCN**, not
lossless PFC.

## What this means per transport

| Transport | Behavior on ECN/DCQCN | Notes |
|---|---|---|
| **MoRI** (lossless-PFC-designed) | Hits a congestion wall at c≥4 | ECN backoff shrinks cwnd faster than recovery loop grows it. Designed for PFC where drops never happen; ECN-marked traffic is treated as a failure path. |
| **Mooncake** (Python, chunked register/transfer/deregister) | Works | Python-level chunking naturally back-pressures against ECN slowdowns. Saturates at c=32 on a different limit (ionic QP-setup queue), not the network. |
| **RIXL/UCX** (C++ transfer pipeline) | Works | UCX's C++ scheduling is congestion-aware; rides ECN cleanly. Saturates at compute, not transport (730 tok/s @ c=32, 320/320 success). |

## Why this reframes the MoRI vs Mooncake architectural finding

The fork's bench cluster runs lossless PFC, which is why MoRI wins
there (178 / 672 / 2196 tok/s at c=4/16/128). On AAC1's lossy
ECN/DCQCN, Mooncake wins (527 tok/s @ c=16, 160/160 success).

**The right transport depends on the network fabric, not which is
"better in absolute terms."** Both work on the same Dynamo + SGLang
+ AMD MI355X stack.

## The full traceback under MoRI failure

The c≥4 crash on AAC1 surfaces as:

```
… ibverbs.cpp:168  Connection timed out
… scheduler_dp_attn_mixin.py:93  unknown parameter type
```

The Python error is **secondary** — the root cause is MoRI's ibverbs
transport hitting the ECN backoff wall, which cascades through DP-Attn's
`all_gather` collective into the PyTorch "unknown parameter type"
exception. Confirmed by reading the NIC counters above (798M cwnd shrinks
vs 5.7M grows is unambiguous congestion-collapse).

## Path forward to reach fork's MoRI numbers on AMD

Three options, in increasing effort order:

1. **PFC config request to AAC1 cluster admins.** Cluster-admin
   operation spanning NIC drivers, switch ACLs, and possibly DCB config.
   Writeup at `~/Documents/aac1_pfc_admin_request.md` (AMD-internal).
   This is the cheapest path if the admins are willing.
2. **Move to a different cluster** with lossless PFC already configured.
   Tensorwave's MI300X box has the right config; we haven't tried it for
   MoRI yet because our DSR1 weights are AAC1-cached.
3. **MoRI source patches.** Add ECN/DCQCN-aware backoff to MoRI's
   congestion controller. Largest effort; only worth doing if option 1
   is permanently blocked AND we need MoRI specifically (rather than
   accepting RIXL/Mooncake as the AMD-native answer).

## Why we ship Mooncake as the primary AAC1 disagg result

It works. 527 tok/s @ c=16 with 160/160 success is the headline.
MoRI c=1 succeeds as an integration-works datapoint but isn't the
production answer on this cluster. Once PFC is enabled (option 1
above), this conclusion may flip — we'd re-run the matrix and update
[`findings.md`](findings.md).
