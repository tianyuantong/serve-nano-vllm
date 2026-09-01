# 上游基线

```
UPSTREAM_SHA = bb823b3e06983d71485a8e1f23715ebd87d98ef8
DATE         = 2026-04-26 13:10:12 +0800
SUBJECT      = Merge pull request #218 from GeeeekExplorer/chunked-prefill-refactor
UPSTREAM     = https://github.com/GeeeekExplorer/nano-vllm
FORK         = https://github.com/tianyuantong/serve-nano-vllm
```

**所有实验 manifest 必须记录这个 SHA。**本项目不跟随上游浮动的 `main`。

## 分支布局

```
upstream/main @ UPSTREAM_SHA
  ├── fix/tp-rendezvous   仅含 TP rendezvous / IPC 命名修复 -> 上游 PR
  └── nanoserve           项目主分支
        ├─ tag: base-common   = + greedy + prefix-cache 开关 + metrics + bugfix
        ├─ tag: scheduler     = + ExecutionMode/StepPlan + decode-first chunked prefill
        └─ tag: speculative   = + n-gram proposer + VERIFY + KVReservation
```

所有 headline 对比只在 `base-common` 及以上的 tag 之间进行。
`upstream` 与 `base-common` 无法做性能对比 —— 上游 `assert temperature > 1e-10`
禁止 greedy，两侧 token 流与长度不同。instrumentation 开销用
`base-common` 的 metrics on/off 测量。
