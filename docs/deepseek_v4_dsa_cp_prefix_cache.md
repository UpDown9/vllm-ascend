# DeepSeekV4 DSA CP 与 Prefix Cache 逻辑

## 1. 阅读结论

本文只讨论 DeepSeekV4 开启 `enable_dsa_cp=True` 后，prefix cache 在本地 KV cache 和 MooncakeStore 中的行为。

默认前提：

```text
enable_dsa_cp = True
prefill_context_parallel_size = 1
decode_context_parallel_size = 1
```

结论先放在前面：

1. `enable_dsa_cp` 是 DeepSeekV4 attention 内部的执行优化，不是 vLLM PCP/DCP。
2. 只开启 `enable_dsa_cp` 不会改变 `PCP/DCP` 参数，也不会把 prefix cache 变成 PCP/DCP 分片语义。
3. 当前工程中，prefix cache 仍由 KV cache coordinator 和可选 MooncakeStore 管理。
4. DeepSeekV4 有多类 cache group，主要包括 C1、C4、C128 和 SWA/SlidingWindowMLA 相关 group。
5. MooncakeStore key 会区分 `group` 和 `cache_family`，所以 C1/C4/C128 不会互相覆盖。
6. MooncakeStore external lookup 当前主要用 C1 dense group 做 gate；load 阶段再尝试加载 metadata 中的所有 group。
7. 如果 C1 命中但 C4/C128 部分缺失，load 会产生 invalid block，scheduler 回退并重算缺失 suffix。
8. DSA CP 影响 forward 内部如何切 token、算 attention、还原输出，不改变 prefix cache 的 block hash/key 语义。

## 2. 两条容易混淆的路径

### 2.1 当前工程的 `enable_dsa_cp`

当前工程里的 DSA CP 路径可以理解为：

```text
request tokens
  -> block hashes
  -> prefix cache lookup/load/save
  -> 本地 KV/cache blocks 可用
  -> DeepSeekV4 forward
  -> DSA CP attention 内部做 token 切分、跨 rank 计算和输出还原
```

也就是说，prefix cache 在 attention forward 之前已经完成匹配和加载。DSA CP 不负责决定哪些 cache block 命中。

相关入口：

- `vllm_ascend/utils.py:1345`: `enable_dsa_cp()`
- `vllm_ascend/attention/dsa_v1.py:189`: 选择 `AscendDSACPMetadataBuilder`
- `vllm_ascend/attention/dsa_v1.py:208`: 选择 `AscendDSACPImpl`
- `vllm_ascend/models/deepseek_v4.py:731`: DeepSeekV4 attention 初始化读取 `enable_dsa_cp`

### 2.2 后续迁移 Geneva CP 后的路径

如果未来把 `/home/xujiuxu/code/omniinfergeneva2` 仓的 DeepSeekV4 CP 方案迁移进来，语义会多一个变化：wincache 可能变成 CP rank 局部 ownership。

但需要注意：

```text
CP rank 局部拥有 wincache block
  != MooncakeStore lookup 要按 CP rank 匹配
```

只要 save 到 MooncakeStore 的 value 是完整 cache block，lookup 仍然按逻辑 block hash 匹配，不需要把 CP rank 放进 key。

## 3. DeepSeekV4 多 Cache Group

DeepSeekV4 不是一个普通的单 KV cache group 模型。当前实现会根据 MLA attention spec、`compress_ratio` 和 SlidingWindowMLA spec 生成多个 cache group。

逻辑上主要有：

```text
C1    : dense/full MLA cache family
C4    : C4 compressor/indexer 相关 cache family
C128  : C128 compressed cache family
SWA   : sliding-window 相关辅助 cache group
```

最终 group 组织大致是：

```text
[full_mla_group, full_mla_c128_group, *swa_mla_groups]
```

相关代码：

- `vllm_ascend/models/deepseek_v4.py:116`: `AscendCompressorStateCache.get_kv_cache_spec`
- `vllm_ascend/models/deepseek_v4.py:147`: `AscendDeepseekV4IndexerCache.get_kv_cache_spec`
- `vllm_ascend/models/deepseek_v4.py:186`: `AscendDeepseekV4SWACache.get_kv_cache_spec`
- `vllm_ascend/models/deepseek_v4.py:780`: layer 级 `compress_ratio`
- `vllm_ascend/patch/platform/patch_kv_cache_utils.py:61`: KV cache specs 分组
- `vllm_ascend/patch/platform/patch_kv_cache_utils.py:95`: 构建最终 KV cache groups

理解这个点很关键：prefix cache 命中不是只看一份 KV cache，而是会涉及多个 group/family。

## 4. 不接 MooncakeStore 时的 Prefix Cache

不接 MooncakeStore 时，只走本地 KV cache manager。

流程：

```text
new request
  -> scheduler 拿到 request.block_hashes
  -> HybridKVCacheCoordinator 按 group 查询本地 prefix cache
  -> 找到每个 group 的 hit blocks
  -> 收敛出本次可复用的 prefix hit length
  -> KV cache manager 基于 hit blocks 创建 KVCacheBlocks
  -> 未命中的 suffix 正常计算
  -> DeepSeekV4 DSA CP 在 forward 内部处理切分和还原
```

关键点：

- block hash 仍然是逻辑 token block 的 hash。
- DSA CP 不改变本地 prefix cache lookup 的 hash 语义。
- 本地命中表示本地 KV/cache blocks 已经可用；attention 只是消费它们。

相关代码：

- `vllm_ascend/core/recompute_scheduler.py:423`: scheduler 查询本地 cached tokens
- `vllm_ascend/core/recompute_scheduler.py:433`: 调用 `find_longest_cache_hit_per_group`
- `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:326`: hybrid cache hit 逻辑
- `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:416`: DeepSeekV4 多 full-attn group 的 truncate 逻辑

## 5. 接 MooncakeStore 时的 Prefix Cache

接入 MooncakeStore 后，本地 prefix cache 之外会增加 external lookup/load/save。

整体流程：

```text
new request
  -> 本地 prefix cache lookup
  -> AscendStore scheduler 计算还需要 external lookup 的 token_len
  -> lookup 请求携带 token_len、block_hashes、kv_cache_group_ids
  -> pool worker 在 MooncakeStore 中检查 key 是否存在
  -> external 命中后 scheduler 创建 LoadSpec
  -> scheduler 分配本地 block
  -> pool worker 从 MooncakeStore load cache blocks 到本地 KV cache
  -> load 失败的 block 记录为 invalid_block_ids
  -> scheduler 必要时截断 computed tokens 并重算失败 suffix
  -> compute 后按 group 保存 cache 到 MooncakeStore
```

Scheduler 侧代码：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:45`: 初始化 AscendStore scheduler
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:55`: hybrid KV cache 使用所有 group id
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:224`: `get_num_new_matched_tokens`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:277`: 创建 `LoadSpec`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:350`: 构造 connector metadata

Lookup RPC 代码：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:294`: lookup server 请求循环
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:301`: 调用 `pool_worker.lookup_scheduler`

## 6. MooncakeStore Key 语义

MooncakeStore 看到的是字符串 key，不是裸 token hash。

当前 key 包含：

```text
model
pcp rank
dcp rank
head_or_tp_rank
pp rank
group id
cache role
cache family
chunk hash
```

格式大致是：

```text
model@pcp{pcp}@dcp{dcp}@head_or_tp_rank:{rank}@pp_rank:{pp}
  @group:{group_id}@cache_role:{role}@cache_family:{family}@{chunk_hash}
```

因此同一个 `chunk_hash`，在不同 cache group/family 下是不同 key：

```text
C1 key   = ... @group:{c1_group}   @cache_family:c1   @{hash}
C4 key   = ... @group:{c4_group}   @cache_family:c4   @{hash}
C128 key = ... @group:{c128_group} @cache_family:c128 @{hash}
```

所以 MooncakeStore 不是“用一个 hash 匹配 DeepSeekV4 所有 cache”。它会通过 `group` 和 `cache_family` 区分 C1、C4、C128。

相关代码：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:61`: `PoolKey.to_string`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:121`: 根据 compress ratio 推断 cache family
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:166`: 推断每个 group 的 cache family
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:228`: 基于 hash 构造 key

## 7. Lookup: 为什么用 C1 做 Gate

当前 MooncakeStore external lookup 没有要求 C1/C4/C128 全部严格同时命中。它会先过滤 group list，通常只保留 dense C1 group 作为 lookup gate。

流程：

```text
kv_cache_group_ids = all groups
  -> _get_lookup_gate_group_ids
  -> 通常保留 dense C1 groups
  -> 构造 C1 keys
  -> m_store.exists(C1 keys)
  -> 找连续命中的最大 prefix
  -> 返回 external hit token length
```

这样做的原因是：

- C128 key stream 更稀疏，如果把 C128 也作为严格 gate，可能让本来 C1 可复用的请求变成 0 hit。
- C4 group 在当前 connector path 中存在 TP-sharded key stream 现象，不适合作为统一 gate。
- block size 不是基础 dense block size 的 group 也不适合作为 gate。

因此可能出现：

```text
C1 hit length > C4 hit length
C1 hit length > C128 hit length
```

这是当前设计允许的，不是 DSA CP 引入的问题。

相关代码：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:924`: `_get_lookup_gate_group_ids`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:936`: `_is_lookup_gate_group`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:970`: `lookup_scheduler`

## 8. Load: 命中后会尝试加载哪些 Cache

lookup 只用 gate groups 判断“外部最长命中 prefix”。但 load 阶段会根据 request metadata 尝试加载配置的 groups。

流程：

```text
LoadSpec exists and can_load=True
  -> for each load_group_id
     -> 生成 group-specific keys
     -> 生成本地 KV cache 目标地址
  -> m_store.get(key_list, addr_list, size_list)
  -> 如果部分 key load 失败
     -> 记录 invalid block ids
     -> scheduler 截断受影响 request 的 computed tokens
     -> missing/invalid suffix 重新计算
```

这就是 C1 命中多于 C4/C128 时的正确性兜底。

相关代码：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:463`: `start_load_kv`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:511`: 遍历 `load_group_ids`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:518`: 构造 key 和 block id
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:557`: 调用 `m_store.get`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:558`: 记录失败 block
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:586`: 暴露 failed block ids
- `vllm_ascend/core/recompute_scheduler.py:790`: scheduler 处理 invalid blocks

## 9. Save: 计算完成后如何写 MooncakeStore

compute 完成后，请求 metadata 会按 group 保存到 MooncakeStore。

流程：

```text
computed request/chunk
  -> ReqMeta 记录 block ids by group
  -> for each group
     -> 根据 token range 和 block hashes 生成 group-specific keys
     -> key 中带 group/cache_family
     -> 跳过已经存在的 key
     -> put 缺失的 key/value 到 MooncakeStore
```

TP 在 save 阶段可以作为上传任务分摊维度。代码可以按 TP rank 对完整 key list 做 stride 分片：

```python
starts = starts[self.tp_rank % self.put_step :: self.put_step]
ends = ends[self.tp_rank % self.put_step :: self.put_step]
keys = keys[self.tp_rank % self.put_step :: self.put_step]
block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]
```

这个分片表示“谁负责上传哪些 key”，不一定表示“这个 TP rank 的逻辑 cache 只有这些 key”。

相关代码：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:699`: `ReqMeta.from_request_tracker`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:709`: metadata 携带所有 allocated group ids
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:269`: save thread request handler
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:278`: 遍历 `req_meta.kv_cache_group_ids`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:291`: 生成 group-specific keys
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:303`: put 阶段按 TP 做 strided key sharding

## 10. DSA CP 与 TP 的边界

在本文前提下，TP 不意味着每个 TP rank 只有一份不同的逻辑 attention cache。

需要区分三件事：

```text
Runtime compute:
  DSA CP/TP 可以切分 Q 或 attention 计算。

MooncakeStore save:
  TP rank 可以分摊上传完整 key list 中的一部分 key。

Prefix cache identity:
  cache block 仍按 group/family/hash 匹配，不因为 DSA CP 自动变成 PCP/DCP-sharded cache。
```

load 阶段，每个 worker 会根据 request/group 构造自己需要的 key list，并调用 `m_store.get`。当前代码会按 TP rank 旋转 key/address/size list，用于顺序和负载均衡，但它表达的仍是该 worker 要加载的逻辑 cache 范围。

相关代码：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:538`: 按 TP rank 旋转 key/address/size list
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:557`: 调用 `m_store.get`

## 11. 当前实现下的 FAQ

### 11.1 开启 `enable_dsa_cp` 会影响 PCP/DCP 吗

不会。

`enable_dsa_cp=True` 只选择 DeepSeekV4 DSA attention 的内部 CP 执行路径。`PCP=1、DCP=1` 时，prefix cache 仍按非 PCP/DCP 的本地或外部 KV pool prefix cache 理解。

### 11.2 C1 命中是否可能比 C4/C128 更多

可能。

当前 MooncakeStore lookup 使用 C1 dense group 做 gate。如果 C4 或 C128 某些 key 缺失，load 阶段会记录 invalid blocks，scheduler 会回退并重算缺失 suffix。

### 11.3 MooncakeStore 是否用一个 hash 匹配所有 DeepSeekV4 cache

不是。

相同 chunk hash 会被包装成带 `group` 和 `cache_family` 的完整 key。C1、C4、C128 是不同 key space。

### 11.4 TP rank 上传的 key 是否一定代表 TP 语义分片

不一定。

当前 save path 可以按 TP rank 分摊上传 key，但这更像上传任务拆分。对于 replicated attention cache，不能简单理解为“每个 TP rank 只拥有自己上传的那部分逻辑 cache”。

## 12. 引入 Geneva CP 后的 MooncakeStore 适配方案

本节讨论另一个目标方案：如果把 `omniinfergeneva2` 仓的 DeepSeekV4 CP 方案迁移到当前工程，并采用“方案 2”的 MooncakeStore key 语义。

这里的核心假设是：DeepSeekV4 attention cache 是 replicated cache，不是 TP/head 维度语义分片 cache。

### 12.1 与 PD `mooncake_hybrid_connector` 分开看

MooncakeStore/AscendStore 和 `mooncake_hybrid_connector` 是两条路径：

```text
MooncakeStore / AscendStore:
  prefix cache lookup/load/save
  按 block hash 和 PoolKey 匹配历史 cache

mooncake_hybrid_connector:
  PD P-to-D KV transfer
  按 request metadata、remote_block_ids、local_block_ids 和远端地址传输
  不使用 MooncakeStore key/hash 做 prefix 匹配
```

所以同时配置 `store` 和 `mooncake_hybrid_connector` 时：

- MooncakeStore 负责跨请求 prefix cache 复用。
- `mooncake_hybrid_connector` 负责当前请求从 P 节点向 D 节点交接 KV cache。

本节只讨论 MooncakeStore prefix cache，不讨论 PD 传输。

### 12.2 方案 2: replicated attention cache 使用统一 logical TP key

当前 `PoolKey` 中包含 `head_or_tp_rank`：

```text
model@pcp{pcp}@dcp{dcp}@head_or_tp_rank:{rank}@pp_rank:{pp}
  @group:{group_id}@cache_role:{role}@cache_family:{family}@{chunk_hash}
```

如果 DeepSeekV4 attention cache 是 replicated cache，那么真实 TP rank 不是 cache 内容的语义维度。

如果继续把真实 TP rank 写入 key，会有两个问题：

- 同一个逻辑 cache block 被拆进多个 TP namespace。
- load 侧必须知道保存时哪个 TP rank 上传过该 block，否则要枚举 producer TP rank key。

方案 2 的做法是：对 DeepSeekV4 replicated attention cache 使用统一 logical rank。

```text
真实 TP rank:
  tp_rank = 0, 1, 2, ...

MooncakeStore key:
  head_or_tp_rank = 0
  或 head_or_tp_rank = replicated sentinel
```

这样 key identity 变成：

```text
model + pcp + dcp + logical_attention_cache_rank + pp_rank
  + group + cache_role + cache_family + chunk_hash
```

而不是：

```text
model + pcp + dcp + real_tp_rank + pp_rank
  + group + cache_role + cache_family + chunk_hash
```

注意：这个规则只适用于 DeepSeekV4 replicated attention cache。其他真实 TP/head-sharded cache 仍应保留真实 `head_or_tp_rank`。

### 12.3 Geneva CP 后 Save 怎么做

Geneva CP 后，wincache 在 CP rank 间按完整 cache block 边界切分。保存到 MooncakeStore 时推荐这样做：

```text
每个 CP rank 只 put 自己拥有的完整 wincache blocks
所有 CP rank put 完后
MooncakeStore 中形成完整 prefix block 集合
```

示例：

```text
prefix blocks: B0 B1 B2 B3 B4 B5 B6 B7
CP size = 2

cp_rank0 owns: B0 B1 B2 B3
cp_rank1 owns: B4 B5 B6 B7

MooncakeStore after save:
  key(B0) -> complete cache(B0)
  key(B1) -> complete cache(B1)
  ...
  key(B7) -> complete cache(B7)
```

关键约束：

- CP rank 不需要进入 `PoolKey`。
- block hash 已经表达逻辑 token block 的身份。
- 每个 put 的 value 必须是完整 cache block。
- 不允许保存 block 内部的局部 token 或局部 head。
- replicated attention cache 的 `head_or_tp_rank` 使用统一 logical key。

非 wincache cache 的处理：

- compressor cache、indexer cache、C4/C128 等仍按当前 cache group/family 保存。
- 如果这些 cache 在计算语义上仍是完整 block，则可以沿用现有 group/family key。
- 如果未来某类 cache 也被 CP 切分，需要单独定义 owner/range 语义，不能隐式套用 wincache 规则。

### 12.4 Geneva CP 后 Lookup 怎么做

Lookup 阶段不需要因为 CP owner 做额外适配。

原因是 MooncakeStore 的匹配对象仍是逻辑 prefix blocks：

```text
request tokens
  -> block hashes
  -> group/cache_family-specific PoolKey
  -> m_store.exists(keys)
```

只要 save 阶段保证每个 key 对应完整 block，lookup 不需要知道该 block 当初由哪个 CP rank 上传。

因此 lookup 仍然可以保持：

```text
all metadata groups
  -> lookup gate groups
  -> usually C1 dense group
  -> exists(C1 keys)
  -> return external hit length
```

CP 不进入 lookup gate 条件。CP 只影响 save 阶段哪个 rank 负责 put 哪些 complete blocks。

### 12.5 Geneva CP 后 Load 怎么做

Load 命中后，每个需要使用 attention cache 的 compute rank 都应该得到自己计算所需的完整 cache 视图。

如果 DeepSeekV4 attention cache 是 replicated cache，则即使开启 TP，每个 rank 需要的仍是完整 attention cache：

```text
rank0 load B0..B7
rank1 load B0..B7
...
```

推荐第一版 load 语义：

```text
for each runtime rank:
  根据 request block hashes 构造统一 logical key
  从 MooncakeStore load 完整命中 prefix blocks
  填充本 rank 本地 KV/cache blocks
```

也就是说：

```text
save 可以由不同 CP/TP rank 分摊 put
load 不能只加载当前 rank 曾经上传过的那部分 key
load 应面向当前 rank 计算需要的完整 cache 构造 key list
```

可选优化是“两阶段 load”：

```text
每个 rank 只从 store load 一部分 blocks
  -> rank 间 allgather / alltoall 汇聚
  -> 每个 rank 得到完整 cache
```

但这条路径复杂度高，第一版不建议做。它需要处理：

- block 到 loader rank 的分配。
- 额外通信把 blocks 汇聚到每个 compute rank。
- load 失败后的 invalid block 广播和一致回退。
- 多 cache group 的 hit/failed 状态对齐。

### 12.6 Geneva CP 后多 Cache Group 怎么处理

Geneva CP 不改变 DeepSeekV4 多 cache group 的匹配方式。

MooncakeStore 仍通过 `group` 和 `cache_family` 区分不同 cache：

```text
C1 key:
  group:{c1_group}@cache_family:c1@hash

C4 key:
  group:{c4_group}@cache_family:c4@hash

C128 key:
  group:{c128_group}@cache_family:c128@hash
```

仍然可能出现：

```text
C1 hit length > C4/C128 hit length
```

处理方式保持当前设计：

- lookup 使用 C1 dense group 作为 gate。
- load 阶段尝试加载 metadata 中需要的所有 group。
- 某些 group load 失败时记录 invalid blocks。
- scheduler 对 invalid suffix 回退并重算。

不要把 C1/C4/C128 合并成一个 key，也不要假设一个 family 命中代表所有 family 都命中。

### 12.7 推荐修改点

如果按方案 2 落地，MooncakeStore 相关代码建议按以下方向修改。

#### 12.7.1 统一 replicated attention cache key

对 DeepSeekV4 replicated attention cache，生成 `PoolKey` 时使用：

```text
head_or_tp_rank = 0
```

或明确 sentinel：

```text
head_or_tp_rank = replicated
```

不能让真实 TP rank 进入 replicated attention cache 的 key identity。

#### 12.7.2 Save 阶段支持 CP block ownership

保存 wincache 时，key list 按当前 CP rank 拥有的完整 blocks 过滤：

```text
all computed blocks
  -> filter blocks owned by current cp_rank
  -> put owned complete blocks
```

过滤必须基于 cache block 边界，不能基于 block 内 token 范围。

#### 12.7.3 Load 阶段按完整 cache 需求加载

load 侧不按 CP owner 过滤 key。每个需要完整 attention cache 的 rank 都构造完整命中 prefix 的 key list：

```text
matched prefix blocks
  -> all block keys
  -> m_store.get(all keys)
  -> fill local KV/cache blocks
```

#### 12.7.4 Lookup 阶段保持逻辑 block hash 匹配

lookup 阶段不引入 CP rank 条件：

```text
block_hash + group/cache_family + logical replicated rank
```

这就是 cache identity。

#### 12.7.5 失败处理沿用 invalid block 回退

如果某些 CP rank 没有成功 put 自己拥有的 blocks，后续可能出现：

```text
C1 gate hit
load missing block
```

这时仍应沿用当前 invalid block 机制，让 scheduler 截断并重算缺失 suffix。

## 13. Geneva CP 方案复杂度

推荐第一版：

```text
save:
  CP rank 只保存自己拥有的完整 wincache blocks

lookup:
  不感知 CP，继续按 logical block hash 匹配

load:
  每个 compute rank 从 store 直接加载完整命中 cache

key:
  DeepSeekV4 replicated attention cache 使用统一 logical head_or_tp_rank
```

复杂度评估：

```text
统一 replicated attention cache key:
  中等
  需要只影响 DeepSeekV4 replicated cache，不能误伤真实 TP/head-sharded cache。

save 阶段 CP owner 过滤:
  中等
  需要从 Geneva CP metadata 或等价结构拿到 block ownership。

lookup 阶段:
  低
  理论上不需要引入 CP 维度。

load 阶段每 rank 直接加载完整 cache:
  低到中等
  逻辑简单，但 store 读流量会放大。

load 阶段分片加载再 allgather:
  高
  涉及 loader 分配、通信、失败一致性和多 group 对齐，不建议第一版实现。
```

## 14. 最终总结

当前工程中，`enable_dsa_cp=True` 后的 prefix cache 可以这样理解：

```text
prefix cache identity:
  仍然是 block hash + group/cache_family + PoolKey 维度

DSA CP:
  只影响 DeepSeekV4 attention forward 内部怎么切分和计算

MooncakeStore lookup:
  当前主要用 C1 dense group 做 gate

MooncakeStore load:
  尝试加载 metadata 中的所有 group，失败则 invalid block 回退

MooncakeStore save:
  按 group/family 保存 computed blocks，TP 可用于上传任务分摊
```

如果后续引入 Geneva CP，第一版推荐：

```text
CP 只影响 wincache save owner
lookup 不引入 CP rank
load 侧每个 compute rank 加载完整需要的 replicated attention cache
DeepSeekV4 replicated attention cache key 不使用真实 TP rank
```

## MooncakeStore 新 CP 适配

新 CP 下，DeepSeekV4 的运行时 cache 语义是：

```text
SWA / wincache : CP rank 本地 owner cache，只保存本 rank 有效 token，不保存 halo
C4 / C128      : 每层通过 all-gather 后恢复为全局可见 cache
state          : 每个 request tail owner 产生，随后 broadcast 到需要的 rank
```

因此 MooncakeStore 的适配重点在 put：

1. 仅对 SlidingWindow/SWA group 做 CP owner 过滤。
2. owner range 按“per-request 128 对齐后 flatten padded unit 流”的连续 CP 切分计算。
3. 只有完整落在本 rank owner range 内的 block 才 put。
4. halo token 不 put。
5. 如果出现跨 owner 边界的 partial block，当前实现跳过并打印 warning；这要求上游 put chunk 与 128-token owner range 对齐。

lookup/load 当前可以复用原逻辑，前提是：

1. lookup gate 仍使用 dense C1 这类全局完整 group，不使用 SWA/wincache group 作为命中完整性的唯一依据。
2. load 读取的是完整 block value；没有引入 partial value slicing。
3. SWA/wincache 的 halo 在 forward 内按当前 chunk 重新 gather/assemble，不依赖 MooncakeStore 保存 halo。

如果后续要支持 partial block put，那么 lookup/load 都需要新增 block 内 token validity 或 value slicing 逻辑；当前方案没有做这件事。
