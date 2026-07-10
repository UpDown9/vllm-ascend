# DeepSeekV4 DSA CP Local Cache 改造方案

本文记录当前分支基于 `/home/xujiuxu/code/omniinfergeneva2` 的 DeepSeekV4 CP 方案改造落点。目标是只讨论 prefill，不覆盖 decode。

## 1. 目标

当前 `vllm-ascend` 的 DSA CP 已经把 Q 和 sparse attention 查询切到本地 rank，但 cache 更新路径仍然先 all-gather 出完整 hidden state，再按完整 token 流计算并写入 SWA、indexer、compressor cache。

改造目标：

- 保持现有 DSA CP 逻辑作为默认路径，不改变线上默认行为。
- 通过显式参数或环境变量控制 DeepSeekV4 DSA CP 走老 CP 方案还是新 local-cache CP 方案。
- CP64 只切 64 份，不使用 Geneva 原始 `2 * cp_size` zigzag。
- 每个 CP rank 只计算自己负责的 query segment。
- SWA/window cache 只计算 local rank 的有效 token，但要能读取前序 128 token halo。
- C4A/indexer 只计算 local rank 的有效压缩 token，但要能读取前序 4 token overlap。
- compressor/indexer 需要 full-cache 视图时，通过 all-gather 汇聚各 rank 的 local compressed 结果，而不是每个 rank 重算全量 hidden state。
- prefix cache 的逻辑 hash/key 语义不改，仍按完整逻辑 token block 管理；CP local ownership 只影响 forward 内部计算和写 cache 的位置。

## 1.1 兼容开关原则

本次改造必须是增量修改，不能直接替换当前 CP 实现。默认路径继续走当前分支已有的 DSA CP 行为：

```text
old DSA CP:
  Q/local attention 使用 local hidden
  cache 路径 all-gather full hidden
  SWA/indexer/compressor 按 full hidden 计算
```

新方案只在显式开启时生效：

```text
new DSA local-cache CP:
  Q/local attention 使用 local hidden
  SWA 使用 local hidden + 128-token halo
  indexer/compressor 使用 local hidden + overlap
  必要时 all-gather local compressed/cache 结果恢复 full visible cache
```

开关来源建议优先使用模型/运行参数；如果工程上短期更容易落地为环境变量，则必须按当前仓库规范集中定义在 `vllm_ascend/envs.py`，不要在 `dsa_cp.py` 或 runner 中硬编码 `os.getenv()`。

建议语义：

```text
enable_dsa_cp_local_cache = false  # 默认 false，保持老 CP
```

若使用环境变量，可设计为：

```text
VLLM_ASCEND_ENABLE_DSA_CP_LOCAL_CACHE=0  # 默认 0，老 CP
VLLM_ASCEND_ENABLE_DSA_CP_LOCAL_CACHE=1  # 新 local-cache CP
```

该开关只控制 DeepSeekV4 DSA prefill CP 的 cache 计算路径，不改变 `cp_size` 获取方式。对于当前工程的 DeepSeekV4 DSA CP，切分 size/rank 沿用 `dsa_cp.py` 修改前逻辑，从 TP group 获取：

```text
cp_size = get_tp_group().world_size
cp_rank = get_tp_group().rank_in_group
```

因此 DeepSeekV4 CP64 场景要求 DSA CP 路径下 `get_tp_group().world_size == 64`。本 local-cache 方案不从 `prefill_context_parallel_size * decode_context_parallel_size` 派生 owner rank。

代码分流原则：

- metadata builder 保留老 metadata 构造逻辑；新 `DSACPLocalCachePlan` 只在开关开启时构造。
- `_forward()` 保留当前 full hidden cache 路径；新 SWA/indexer/compressor local-cache 路径只在开关开启时进入。
- prefix cache、external cache、block hash、MooncakeStore key 语义不受开关影响。
- 开关关闭时，数值、通信、cache 写入行为应与当前分支一致。
- 开关开启但遇到未支持场景时，第一版可以显式 fallback 到老 CP 路径，不能静默写入不完整 cache。

## 2. 当前分支现状

关键文件是 `vllm_ascend/attention/context_parallel/dsa_cp.py`。

当前 `_forward()` 的结构：

```text
hidden_states_local
  -> maybe_all_gather_and_maybe_unpad()
  -> hidden_states_cache = full hidden
  -> Q 使用 hidden_states_local
  -> SWA KV 使用 hidden_states_cache 写全量
  -> indexer compressor 使用 hidden_states_cache 写全量
  -> C128/C4 compressor 使用 hidden_states_cache 写全量
  -> sparse attention 使用 local Q + local metadata
```

代码落点：

- `dsa_cp.py:1077`: `maybe_all_gather_and_maybe_unpad(hidden_states_local, need_gather_q_kv)` 得到 full hidden。
- `dsa_cp.py:1091`: `hidden_states_cache = hidden_states[:num_actual_tokens]`。
- `dsa_cp.py:1093-1149`: Q 只使用 `hidden_states_local`，这部分已经是 local 计算。
- `dsa_cp.py:1151-1162`: SWA KV 由 full `hidden_states_cache` 计算并 scatter。
- `dsa_cp.py:1171-1178`: C4 indexer cache 更新传入 full `hidden_states_cache`。
- `dsa_cp.py:1191-1217`: C128/C4 compressor 传入 full `hidden_states_cache`，并使用完整 slot mapping 写 cache。
- `dsa_cp.py:1312-1368`: `_update_indexer_cache()` 内部也按 full `x` 调 compressor 并写 indexer cache。

因此，当前分支不是“cache 全量计算、只更新 CP 切分后的有效部分”，而是大部分 cache 路径仍然按 full hidden 在每个 rank 上重算，再由 slot mapping 决定写入位置。local 切分主要体现在 Q 和 sparse attention 查询侧。

## 3. state_block 获取链路

`torch.ops._C_ascend.compressor` 的 `state_block_table` 来自对应 state cache group 的 block table，而不是在 op 内部生成。

链路如下：

```text
model_runner_v1.py
  -> _get_block_table_and_slot_mapping(kv_cache_gid)
  -> AscendCommonAttentionMetadata.block_table_tensor / slot_mapping
  -> AscendDSACPMetadataBuilder.build()
  -> AscendDSAReqMetadata.block_table
  -> dsa_cp.py compressor(..., state_block_table=req_metadata.block_table)
```

代码落点：

- `vllm_ascend/worker/model_runner_v1.py:2985-3048`: 每个 `kv_cache_gid` 获取对应 `block_table` 和 `slot_mapping`。
- `vllm_ascend/worker/model_runner_v1.py:3078-3109`: 构造 `AscendCommonAttentionMetadata`。
- `vllm_ascend/worker/model_runner_v1.py:3217-3219`: `kv_cache_gid > 0` 时替换为当前 cache group 的 `block_table` 和 `slot_mapping`。
- `vllm_ascend/attention/context_parallel/dsa_cp.py:641-655`: builder 把 `self.block_table[:num_reqs]` 和 `slot_mapping` 放入 `AscendDSAReqMetadata`。
- `vllm_ascend/attention/context_parallel/dsa_cp.py:1201`: C128/C4 compressor 使用 `compressor_kv_state_metadata.req_metadata.block_table`。
- `vllm_ascend/attention/context_parallel/dsa_cp.py:1340`: indexer compressor 使用 `indexer_kv_state_metadata.req_metadata.block_table`。
- `vllm_ascend/worker/block_table.py:31-45`: compressed cache group 会按 `compress_ratio` 缩小 `max_num_blocks_per_req`。
- `vllm_ascend/worker/block_table.py:142-172`: 普通 slot mapping 生成。
- `vllm_ascend/worker/block_table.py:174-202`: compressed/draft slot mapping 生成。

改造后不能简单复用 full slot mapping。需要新增 local owner 写入计划，确保 local compressed token 对应到正确的 state/cache block 和 offset。

### 3.1 当前 compressor slot_mapping 链路

当前 DSA CP 中没有单独的 compressor local slot mapping 构建点。compressor 写 compressed KV cache 时使用的是 metadata builder 中从 full slot mapping 裁剪出的前缀：

```text
AscendCommonAttentionMetadata.slot_mapping
  -> DeviceOperator.format_dsa_slot_mapping()
  -> AscendDSACPMetadataBuilder.self.slot_mapping
  -> _get_slot_mapping_size(input_positions, compress_ratio, ...)
  -> self.slot_mapping[:slot_mapping_size]
  -> compressor_attn_metadata.req_metadata.slot_mapping
  -> DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, slot_mapping)
```

对应代码落点：

- `vllm_ascend/attention/context_parallel/dsa_cp.py:343-344`: 把 `common_attn_metadata.slot_mapping[:num_input_tokens]` format 后写入 `self.slot_mapping`。
- `vllm_ascend/attention/context_parallel/dsa_cp.py:632-635`: 根据 compressed positions 的数量计算 `slot_mapping_size`，再取 `self.slot_mapping[:slot_mapping_size]`。
- `vllm_ascend/attention/context_parallel/dsa_cp.py:1275-1277`: compressor 输出通过 `compressor_attn_metadata.req_metadata.slot_mapping` scatter 到 compressed KV cache。

这条链路在 full hidden cache 路径下成立，因为每个 rank 都按 full token 流生成完整 compressor 输出。但 local-cache CP 中，`compressed_kv` 只包含当前 rank owner 的 compressed token 加 overlap 计算结果，不能继续使用 `self.slot_mapping[:slot_mapping_size]`。否则 local 输出会被写到 full compressed token 流的前缀位置，rank 边界和多 batch/chunk prefill 场景都会错位。

因此后续实现必须新增 `compress_local_slot_mapping`：

- 输入来自 local compressed owner token 的全局 token/position 范围，而不是 full compressed positions 前缀。
- 只包含当前 rank 应写入 cache 的 valid compressed token，overlap/borrowed/pad token 不能写 cache。
- 与 `compress_local_state_block_table`、local `cu_seqlens`、local `compress_sin/compress_cos` 使用同一套 local compressed token 顺序。
- 写 `compress_kv_cache` 时使用 `compress_local_slot_mapping`；all-gather 恢复 full compressed KV 视图时使用独立的 gather/restore index，不能复用 scatter slot mapping。

## 4. 与 omniinfergeneva2 的差异

Geneva 蓝本中 CP metadata 做了完整的 segment plan、window plan、compressor plan 和 all-gather reverse KV。但原始方案有两个不适合当前目标的点：

- 原始 `cp_segment_num = cp_size * 2`。
- 每个 rank 取 front/back 两个 segment，带 zigzag/reverse gather 语义。

本分支目标是不使用 zigzag：

```text
cp_segment_num = cp_size
local_segment_id = cp_rank
每个 rank 每个 request 只拥有一个连续 segment
```

迁移时需要保留 Geneva 的 plan 思路，但删除 zigzag 语义：

- segment plan：从 front/back pair 改成单 segment owner plan。
- window plan：仍保留前序 128 token halo all-gather。
- compressor segment input：保留 local input + overlap 构造能力。
- compressor flag/write plan：保留 local 写入和 invalid/pad mask。
- reverse gather：改成 rank 顺序 all-gather，不做 zigzag reverse 还原。

### 4.1 128-token unit 切分与 padding 策略

当前采用方案 A：每个 request 先独立按 128-token unit 做 logical padding，然后把所有 request 的 padded unit 按调度顺序 flatten 成一条 unit 流，再根据 `cp_size` 给每个 rank 分配连续的 128-token unit。`query_start_loc`、`seq_lens`、KV len 仍然只表示真实 token 长度，不包含 padding。

每个 request 的基本规则：

```text
unit_size = 128
req_len = query_start_loc[i + 1] - query_start_loc[i]
req_units = ceil(req_len / unit_size)
req_padded_len = req_units * unit_size
```

然后把所有 request 的 padded unit flatten 后按 rank 顺序分配连续 unit：

```text
total_units = sum(req_units for all scheduled requests)
base_units_per_rank = total_units // cp_size
extra_units = total_units % cp_size
rank i units = base_units_per_rank + (1 if i < extra_units else 0)
owner range = rank i 在 padded unit 流中的连续 unit range
valid range = owner range 映射回真实 flattened token 后的有效 token range
```

这意味着 rank 间允许 local valid token 数不同，且一个 rank 可以持有多个 request 的 valid piece。logical padding 只参与 owner 计算和必要的通信 shape 对齐，不能进入 attention、slot mapping 或 cache write。

单个 9K request 在 CP64 下的推荐切法：

```text
query_len = 9K = 9216 tokens
unit_size = 128
cp_size = 64

req_units = 9216 / 128 = 72
base_units_per_rank = 72 // 64 = 1
extra_units = 72 % 64 = 8
```

owner 分配：

```text
rank0 - rank7  : 2 units = 256 valid tokens
rank8 - rank63 : 1 unit  = 128 valid tokens
```

这样逻辑计算规模仍是：

```text
72 * 128 = 9216 tokens
```

不能为了让所有 rank 等长而统一提升到 2 units：

```text
64 * 2 * 128 = 16384 tokens
```

后者会把 9K 请求 padding 到 16K，padding 放大过高。

多 batch 时每个 request 只负责 128 对齐，CP owner 在 flatten 后的 padded unit 流上连续切分：

- halo/overlap 不能跨 request 借 token。
- padding token 只用于 owner 计算和临时 shape 对齐，不能写 cache。
- `local_valid_ranges + local_offsets` 描述 compact local hidden；多 batch 下不能再只依赖一个连续 `[local_start, local_end)`。
- `rank_valid_ranges[rank][req]` 是后续 SWA、C4/C128、state plan 的真实 token 边界。

## 5. 新 metadata 设计

建议在 `DSACPMetadata` 中扩展以下字段，或者新增一个 `DSACPLocalCachePlan` 被 `DSACPMetadata` 持有。

必需字段：

```text
segment_start_cpu / segment_end_cpu
segment_start / segment_end
local_positions
local_slot_mapping
local_state_block_table
window_input_start / window_input_end
window_valid_start / window_valid_end
compress_input_start / compress_input_end
compress_valid_start / compress_valid_end
compress_local_slot_mapping
compress_local_state_block_table
gather_recv_counts / gather_recv_offsets
restore_indices
valid_mask
```

语义：

- `segment_start/end`: 当前 rank 真正负责的 query token 范围。
- `window_input_start/end`: 为 SWA 计算准备的输入范围，包含前序最多 128 token halo。
- `window_valid_start/end`: SWA 写 cache 时只写 local segment 的有效 token。
- `compress_input_start/end`: 为 C4/C128 compressor 准备的输入范围，包含前序 overlap。
- `compress_valid_start/end`: compressor 输出中属于当前 rank owner 的有效 compressed token。
- `local_slot_mapping`: local 写 cache 使用，不能直接用 full slot mapping。
- `local_state_block_table`: local compressor 读取/更新 state cache 时使用。
- `gather_recv_counts/offsets`: 汇聚各 rank local compressed KV 后恢复 full-cache 视图。
- `restore_indices`: all-gather 后按原始 token/压缩 token 顺序还原。

### 5.1 多 batch 请求样例

假设：

```text
cp_size = 4
unit_size = 128
window_size = 128
compress_overlap = 4

batch0 query_len = 320
batch1 query_len = 96
```

每个 request 先独立按 128-token unit 做 logical padding，owner 在 flatten 后统一分配：

```text
query_start_loc = [0, 320, 416]
req0 len = 320 -> req_units = ceil(320 / 128) = 3 -> padded_len = 384
req1 len = 96  -> req_units = ceil(96 / 128)  = 1 -> padded_len = 128
total padded units = 4
total padded len = 512
```

padded unit 流按 CP4 连续切分后：

```text
unit0 [0, 128)   -> req0 real [0, 128)       -> rank0
unit1 [128, 256) -> req0 real [128, 256)     -> rank1
unit2 [256, 384) -> req0 real [256, 320)     -> rank2
unit3 [384, 512) -> req1 real [320, 416)     -> rank3
```

compact local hidden 只保留 valid token：

```text
rank0 local_valid_ranges = [(0, 128)]
      local_offsets      = [0]
      local_num_tokens   = 128

rank1 local_valid_ranges = [(128, 256)]
      local_offsets      = [0]
      local_num_tokens   = 128

rank2 local_valid_ranges = [(256, 320)]
      local_offsets      = [0]
      local_num_tokens   = 64

rank3 local_valid_ranges = [(320, 416)]
      local_offsets      = [0]
      local_num_tokens   = 96
```

对应的 window input：

```text
rank0:
  req0 window_input = [0, 128), valid = [0, 128)

rank1:
  req0 window_input = [0, 256), valid = [128, 256)

rank2:
  req0 window_input = [128, 320), valid = [256, 320)

rank3:
  req1 window_input = [320, 416), valid = [320, 416)
```

注意：

- `query_start_loc=[0, 320, 416]` 保持真实长度，不写入 logical padding 后的 `[0, 384, 512]`。
- halo 不能跨 batch/request。`req1` 的前序窗口从 `req1` 起点 320 开始，不能从 `req0` 借。
- padded token 只用于 owner 计算和临时 collective shape 对齐，不能写 slot mapping 或 cache。
- `local_query_start_loc` 是当前 rank compact valid token 的 per-request prefix sum，空 request 长度为 0。

例如 rank0：

```text
local_query_start_loc = [0, 128, 128]
local_seq_lens = [128, 0]
```

例如 rank3：

```text
local_query_start_loc = [0, 0, 96]
local_seq_lens = [0, 96]
```

这里 `local_seq_lens` 表示 local query 在原 request 内可见到的逻辑 KV 长度，不是 local token 数。

### 5.2 chunk prefill 请求样例

假设同一个 request 总长度很长，本轮只调度一个 chunk：

```text
prompt_len = 4096
num_computed_tokens = 2048
num_scheduled_tokens = 512
current chunk logical range = [2048, 2560)
cp_size = 4
window_size = 128
compress_overlap = 4
```

本轮只切 `[2048, 2560)` 这个 suffix chunk，而不是重新切完整 `[0, 2560)`：

```text
rank0 owner = [2048, 2176)
rank1 owner = [2176, 2304)
rank2 owner = [2304, 2432)
rank3 owner = [2432, 2560)
```

window input 需要前序 128 token：

```text
rank0:
  window_input = [1920, 2176)
  valid = [2048, 2176)

rank1:
  window_input = [2048, 2304)
  valid = [2176, 2304)

rank2:
  window_input = [2176, 2432)
  valid = [2304, 2432)

rank3:
  window_input = [2304, 2560)
  valid = [2432, 2560)
```

其中 `rank0` 的 `[1920, 2048)` 已经属于已计算历史，不在本轮 `hidden_states_local` 里。实现上需要明确它的来源不是 KV prefix cache，而是按 request、按 layer 维护的 hidden-state halo cache。

推荐机制：每一层计算结束后，把每个 request 的最后 128 个有效 token 的 layer-output hidden state 广播到所有 CP rank，作为下一层或下一 chunk 的前序 halo。这样第二个 chunk 进入同一层计算时，rank0 可以从该 hidden-state halo cache 拿到 `[1920, 2048)`。

```text
window_input [1920, 2048): 从 hidden-state halo cache 读取
window_input [2048, 2176): 从本轮 rank0 local hidden 读取
valid [2048, 2176): 由 rank0 写 cache
```

需要注意：第 L 层需要的 halo 是第 L 层的输入 hidden state，通常来自第 L-1 层输出后广播保存的 halo；第 0 层输入来自 embedding，需要单独保证 embedding hidden 的 128-token halo 可用，或者能通过 token ids/positions 重新构造。

C4A/compressor overlap 同理，但窗口更短：

```text
rank0 compress_input = [2044, 2176), valid = [2048, 2176)
rank1 compress_input = [2172, 2304), valid = [2176, 2304)
rank2 compress_input = [2300, 2432), valid = [2304, 2432)
rank3 compress_input = [2428, 2560), valid = [2432, 2560)
```

这里 `rank0 compress_input [2044, 2048)` 也来自 hidden-state halo cache，而不是当前 rank 的 local hidden。metadata 需要能区分：

```text
borrowed_from_hidden_state_halo_cache
borrowed_from_previous_rank
owned_by_current_rank
padding_invalid
```

第一版可以不把这些来源拆成多个字段，但 `DSACPLocalCachePlan` 至少要能生成最终的 input tensor、valid mask、local slot mapping 和 local state block table，保证 borrowed token 不被当前 rank 重复写 cache。

## 6. 分路径改造

### 6.1 Q 路径

当前 Q 已经使用 `hidden_states_local`：

```text
dsa_cp.py:1093-1149
```

需要做的主要是把 `_build_local_token_metadata()` 从“flatten token stream 均分”改为“按 request 内 no-zigzag segment plan 切分”，并保证：

- `local_query_start_loc` 对齐 local segment。
- `local_seq_lens` 表示 local query 可见的逻辑 KV 长度。
- `local_cos/local_sin` 使用原始 position，不使用 pad 后错误 position。

### 6.2 SWA/window cache

当前 SWA 用 full hidden：

```text
dsa_cp.py:1151-1162
```

改造为：

```text
local hidden
  + all-gather/neighbor gather 获取前序 128 token halo
  -> wkv/kv_norm/rope
  -> 只 scatter local valid token
```

注意点：

- halo 只参与计算，不写入当前 rank owner 的 cache。
- 对 request 开头不足 128 token 的场景要裁剪。
- batch 内每个 request 的 halo 边界独立，不能跨 request 借 token。
- prefix hit 或 chunk prefill 只计算 suffix 时，halo 可能来自按 request/layer 保存的 hidden-state halo cache，也可能来自其他 rank 本轮 local hidden，需要 metadata 区分；这里不要和 KV prefix cache 混用。

### 6.3 C128/C4 compressor

当前 compressor 用 full hidden：

```text
dsa_cp.py:1191-1217
```

改造为：

```text
local segment hidden + overlap hidden
  -> torch.ops._C_ascend.compressor
  -> 得到 local compressed_kv
  -> scatter local owner 部分
  -> all-gather local compressed_kv
  -> 恢复需要 full-cache 视图的后续输入
```

不能只把 `hidden_states_cache` 改成 `hidden_states_local`，原因：

- compressor 的 `cu_seqlens` 当前是 full query_start_loc，需要替换为 local/borrowed input 对应的 cu_seqlens。
- `compress_sin/compress_cos` 当前按 full compressed positions 构造，需要有 local compressed positions。
- `state_block_table` 当前是 full request block table，需要和 local compressed token owner 对齐。
- `slot_mapping` 当前是 full compressed slot mapping 的前缀截断，不是 local owner mapping；必须新增 `compress_local_slot_mapping`。
- C4/C128 输出数量与原始 token 数不是一一对应，rank 边界附近需要 overlap 和 pad plan。

### 6.4 compressor/indexer state cache broadcast

Geneva 蓝本里 compressed KV 数据和 state cache 的同步不是同一种 collective：

```text
compressed/indexer KV:
  local compressor/indexer output
  -> 过滤 borrowed output，只保留 owner output
  -> allgather 各 rank 的 owner KV update
  -> 接收端使用 metadata 中预构造的 per-rank slot_mapping 调用同一 scatter op
  -> 恢复 full visible compressed/indexer KV cache

compressor/indexer state cache:
  read/clone state_block_table 对应 state blocks
  -> local compressor 更新当前 rank cache 中的 state
  -> 每个 request 找到最后有效 owner rank
  -> 从 owner rank broadcast selected state blocks
  -> 只写回 state_block_table 对应 blocks
```

本方案也需要保留这层语义。原因是 state cache 表示每个 request 当前压缩状态，最终状态只能由该 request 最后有效 token 所在的 owner rank 给出；其它 rank 即使参与了前段 local 计算，也不能用自己的中间 state 覆盖最终 state。

当前已新增 `DSACPStateBroadcastPlan`，并挂在 `DSACPMetadata.state_broadcast_plan` 上：

```text
source_ranks           # shape: [num_reqs]，每个 request 当前 chunk 最后有效 token 的 CP owner；空请求为 -1
local_request_indices  # 当前 rank 需要作为 state broadcast 源的 request index
tail_token_offsets     # shape: [num_reqs]，每个 request 当前 chunk 的 tail token 在 flattened batch 中的 offset；空请求为 -1
state_block_ids        # shape: [num_reqs]，每个 request tail state 对应的 state cache block id；无效为 -1
state_block_indices    # shape: [num_reqs]，每个 request tail state 在 state_block_table 中的列号；无效为 -1
state_valid_mask       # shape: [num_reqs]，该 request 本轮是否需要同步 state block
```

`state_block_ids` 的计算规则是：先用 request tail token 的 request-local position 计算 `state_index = tail_position // compress_ratio`，再计算 `state_block_index = state_index // state_block_size`，最后从当前 cache group 的 `state_block_table[req_idx, state_block_index]` 取 block id。这里的 `state_block_size` 使用当前 builder 传入的 cache group block size，因此 C4 indexer state cache 和 C128 compressor state cache 会按各自的 block size 生成 metadata。`state_block_ids`、`source_ranks` 和 `state_valid_mask` 在 metadata 阶段落到 CPU 小张量，forward 阶段不对设备 tensor 逐个 `.item()`。

当前代码已在 env 开关开启且存在 local plan 时接入真实 selected state block broadcast：

- main compressor state cache：`_forward()` 在 compressor op 和 compressed KV scatter 后调用 `_broadcast_dsa_cp_state_blocks(state_cache, compressor_state_broadcast_plan)`。
- C4 indexer state cache：`_update_indexer_cache()` 在 indexer compressor/scatter 后调用 `_broadcast_dsa_cp_state_blocks(indexer_state_cache, state_broadcast_plan)`。
- broadcast 源 rank 来自 `DSACPStateBroadcastPlan.source_ranks`；实现上通过 `get_tp_group().ranks[group_rank]` 映射到全局 rank，保证 TP/CP group 不是 world group 时源 rank 仍正确。
- 如果 local segment 没有有效 compressed output，则跳过本地 KV scatter，但不提前 return，仍会进入 cache update allgather 和 state broadcast collective，避免 rank 间 collective 序列不一致。

写入流程：

```text
before local compressor:
  cur_state = cache[state_write_block_ids].clone()

after local compressor:
  cur_state = cache[state_write_block_ids].clone()
  selected_state = broadcast_selected_state(
      cur_state,
      owner_ranks=state_owner_rank,
      valid_mask=state_valid_mask,
  )
  cache[state_write_block_ids] = where(valid_mask, selected_state, old_state)
```

注意点：

- broadcast 的是 `state_block_table`/`state_write_block_ids` 选中的 state blocks，不是整个 `cmp_kvcache` 或 `indexer_k_cache`。
- `DSACPStateBroadcastPlan.source_ranks` 对应 Geneva 的 `last_rank_zz`，但本方案无 zigzag，应按 no-zigzag 128-token unit owner 计算。
- 多 batch 时 owner 是 per-request 的，不能用单个 rank 作为整个 batch 的 owner。
- chunk prefill 时，如果某个 request 本轮没有有效 token，则不能更新它的 state block。
- padding token 不能成为 state owner，也不能写 state cache。
- 如果第一版选择 all-gather 后恢复 full visible cache，state cache 仍然需要 selected broadcast；all-gather compressed KV 不能替代 state broadcast。

### 6.4.1 compressed/indexer cache update allgather

当前代码已接入 full visible cache 恢复，但实现不是直接对整个 cache 做 all-reduce，也不是广播整块 cache。原因是 prefix cache/历史 cache 在各 rank 上已经存在，如果对完整 cache 做 SUM 会把历史内容重复累加。

当前实现选择按本轮 owner update 同步：

```text
metadata 阶段:
  DSACPSWAWindowPlan.all_rank_slot_mappings
  DSACPCompressorSlotPlan.all_rank_slot_mappings
  -> 预先记录每个 CP rank owner update 对应的 cache slot

每层 forward:
  本 rank compressor/indexer/SWA 生成 local/overlap output
  -> valid_output_mask 过滤 borrowed output
  -> 本地 scatter owner output
  -> allgather(padded owner KV update)

其它 rank:
  从 allgather buffer 取 source rank 的有效 KV update
  -> 使用 metadata 中 source rank 的 slot_mapping
  -> 调用同一 scatter/quant scatter op
  -> 本地 cache 获得该 source rank owner slots
```

metadata 侧在 `DSACPCompressorSlotPlan.all_rank_valid_output_counts` 中记录每个 CP rank 本轮会产生多少个 owner compressed output，在 `DSACPCompressorSlotPlan.all_rank_slot_mappings` 中记录这些 output 对应的 compressed cache slots。SWA 对应使用 `DSACPSWAWindowPlan.all_rank_valid_token_counts` 和 `DSACPSWAWindowPlan.all_rank_slot_mappings`。forward 阶段只 allgather KV update，不再逐层通信 slot mapping。

由于 allgather 需要固定 shape，forward 会先 allgather 一个很小的 update shape metadata，确定 KV trailing shape 后再 allgather padded owner KV。这个 shape metadata 不包含 slot mapping，也不改变 cache owner 语义。

已接入的位置：

- SWA cache：`_forward()` 在本地 `dsa_kv_compress_scatter` 后调用 `_all_gather_dsa_cp_swa_cache_updates()`。
- main compressor cache：`_forward()` 在本地 `dsa_kv_compress_scatter` 后调用 `_all_gather_dsa_cp_compressed_cache_updates()`。
- C4 indexer cache：`_update_indexer_cache()` 在本地 indexer k/scale/full cache scatter 后调用 `_all_gather_dsa_cp_indexer_cache_updates()`。
- source rank 自己不重复 scatter；它只参与 allgather，本地 cache 已在 allgather 前完成 owner scatter。


### 6.5 C4 indexer cache

当前 `_update_indexer_cache()` 用 full hidden：

```text
dsa_cp.py:1171-1178
dsa_cp.py:1312-1368
```

改造与 C128/C4 compressor 类似，但还要处理 quant/full cache：

```text
local hidden + overlap
  -> indexer compressor
  -> rotate/quant
  -> scatter local indexer_k_cache / indexer_full_cache / scale_cache
  -> broadcast owner indexer update 并在各 rank scatter 恢复 full visible cache
  -> broadcast 每个 request 最后 owner rank 的 indexer state blocks
```

`_indexer_select_topk()` 当前查询侧已经使用 local Q：

```text
dsa_cp.py:1179-1189
```

但它读取的 indexer cache/block table 必须能看到完整历史。因此 local indexer cache 写入后，需要保证所有 rank 后续 sparse attention 查 topk 时能访问完整逻辑 cache，或者 kernel 支持按 owner 分布式读取。

第一版建议仍保留“all-gather 后重建 full indexer/cache 视图”，先保证正确性。

## 7. Prefix Cache 子计划概览

本方案不改变 prefix cache lookup/save 的 key 语义：

- block hash 仍按完整逻辑 token block。
- MooncakeStore key 语义不变，仍沿用当前工程已有 model、rank、pp rank、group、cache role、cache family、chunk hash 等维度；DeepSeekV4 DSA CP 的 local owner rank 不新增 key 维度，切分 rank 取自 TP group。
- C1/C4/C128 仍通过 group/cache_family 区分。

这里的 prefix cache 指 KV/compressor external cache 语义，不包含本方案新增的 hidden-state halo cache。hidden-state halo cache 是 forward 内部为了 local SWA/compressor 计算边界正确性保存的临时 hidden state，按 request 和 layer 管理，生命周期至少覆盖下一层或下一 chunk 的 halo 读取。

但 forward 内部的 cache 写入会从 full-write 变成 owner-write，因此 save 到 external store 前必须确认本地 KV cache 中每个逻辑 block 已经被各 rank owner 写完整。第一版如果采用 all-gather 后写 full cache，则 external store 无需改；如果采用真正分布式 owner cache，则 connector/save/load 也要扩展 CP owner 语义。

建议第一版保持 external store 语义不变：forward 内部 local 算，all-gather 后恢复 full cache 写入或 full visible cache。

Prefix Cache 不再作为独立设计文档维护，而是作为 Local Cache Plan 的一个子计划。完整的 lookup、load、save、MooncakeStore key、CP owner 和失败回退设计见第 13 章。

## 8. 分阶段路线

### Phase 0: 文档和 metadata UT

目标：

- 明确新 local-cache CP 是开关控制的增量路径，默认老 CP 行为不变。
- 新增 no-zigzag segment plan 单元测试。
- 覆盖 CP2/CP4/CP64、多 request、短请求、非 128 对齐、prefix hit suffix 场景。

落点：

- `vllm_ascend/attention/context_parallel/dsa_cp.py`
- `tests/ut/attention/` 或现有 DSA CP 相关 UT 目录。

验收：

- 开关关闭时，metadata、通信、cache 写入行为与当前老 CP 一致。
- 每个 token 只归属一个 rank。
- 所有 rank 的 local segment 拼回后等于原始 token 顺序。
- halo/overlap 不跨 request。

### Phase 1: no-zigzag local Q metadata

目标：

- 在开关开启时替换 `_build_local_token_metadata()` 的 flatten 均分逻辑。
- 开关关闭时继续使用当前 metadata 构造和 full hidden cache 路径。
- 保持当前 full cache 计算不变，仅验证 local Q 和输出还原正确。

落点：

- `dsa_cp.py:521-656`
- `dsa_cp.py:658-743`

验收：

- 开关关闭时，结果与修改前老 CP 一致。
- 开关开启时，CP local query metadata 正确。
- 与当前 full hidden cache 路径结果一致。

### Phase 2: SWA/window local cache

目标：

- 为 SWA cache 增加 128-token halo input plan。
- `dsa_cp.py:1151-1162` 改成 local input 计算，local valid scatter。

当前实现状态：

- `DSACPMetadata` 已新增 `swa_slot_mapping`、`swa_valid_start`、`swa_valid_end`。
- `DSACPMetadata.swa_window_plan` 已记录 per-request 的 `input_ranges`、`valid_ranges`、`halo_ranges`，halo 按 request 边界裁剪，不跨 request 借 token。
- 开关开启时，SWA cache 写入只使用当前 rank owner 的 valid token，对应 `self.slot_mapping[valid_start:valid_end]`。
- 开关开启时，SWA 不再依赖 forward 起始处的 full hidden all-gather；只对当前 rank owner 的 valid token 生成 SWA KV。
- 每个 rank 在 layer forward 开始时通过 allgather 同步本地 slice 尾部最多 128 个 hidden token，供其它 rank 组装需要跨 rank 前序窗口的输入。
- SWA 本身只写当前 rank owner token；前序窗口 token 的 cache 由其 owner rank 写入并通过 `_all_gather_dsa_cp_swa_cache_updates()` 同步到其它 rank。
- pad token 不写 SWA cache；rank 超出 actual token 范围时跳过 SWA scatter。

落点：

- `DSACPMetadata` 新增 window/SWA owner write plan。
- `_forward()` SWA KV 计算段。

验收：

- SWA local valid cache 写入范围与 owner plan 一致。
- pad token 不写 cache。
- 需要 NPU 实机验证 SWA cache 与 full hidden 版本数值一致。
- rank0/request 起点/短请求 halo 裁剪正确。

### Phase 3: C128 compressor local 化

目标：

- local compressor 输入 + overlap plan。
- local compressed slot mapping/state_block_table；`compress_local_slot_mapping` 必须按 local owner compressed token 构建，不能复用 `self.slot_mapping[:slot_mapping_size]`。
- allgather local owner compressed output，并在各 rank scatter 恢复 full compressed KV 视图。
- broadcast 每个 request 最后 owner rank 的 compressor state blocks。

当前实现状态：

- `DSACPCompressorSlotPlan` 已记录 per-request 的 `input_ranges`、`valid_ranges`、`overlap_ranges`。
- 构造 local compressor slot mapping 时，长度按 local compressor input 会产生的 compressed output 对齐。
- overlap/borrowed token 产生的 compressed output 对应 slot mapping 填 `-1`，避免写入 KV cache；owner valid output 才使用真实 compressed cache slot。
- overlap 按 request 边界裁剪，不跨 request 借 token；C4 使用前序 4 token，C128 使用前序 128 token。
- `DSACPCompressorSlotPlan.output_indices` 记录 local compressor output 对应的 full compressed output 下标。
- `DSACPCompressorSlotPlan.compressed_positions` 记录 local compressor output 对应的 compressed RoPE position，规则与现有 full 路径一致：`position + 1 - compress_ratio`。
- `DSACPCompressorSlotPlan.input_indices` 记录 local/overlap compressor 输入 token 在 full hidden 中的下标。
- `DSACPCompressorSlotPlan.input_query_start_loc` 记录 local compressor 输入的 per-request cu_seqlens。
- `DSACPCompressorSlotPlan.request_indices` 和 `start_pos_offsets` 记录 local input ranges 对应的原始 request 行，以及每段 input 在 request 内的起始偏移，用于裁剪 state block table 和修正 `start_pos`。
- forward 已在 env 开关开启且存在 local plan 时，将 compressor op 输入切到 local/overlap hidden。
- local compressor op 使用 `input_query_start_loc` 作为 local `cu_seqlens`，使用 `request_indices` 裁剪 `state_block_table`，并用 `start_pos + start_pos_offsets` 修正每段 local input 的 request 内起点。
- C128/C4 local `compress_sin/compress_cos` 已由 `compressed_positions` 直接生成；未开启 local plan 时仍使用原 full compressed positions 路径。
- local compressor 输出在 scatter 前使用 `valid_output_mask` 过滤 borrowed output，只把 owner output 和真实 slot mapping 传给底层 scatter；overlap/borrowed output 的 slot mapping 为 `-1`，不会写入 KV cache，也不会传给 scatter op。
- 已新增 `DSACPStateBroadcastPlan`，能计算每个 request tail token 的 no-zigzag owner rank，以及当前 rank 需要负责广播的 request 集合。
- local owner compressed output allgather 已接入；各 rank 会从 allgather buffer 取 source rank 的 filtered KV，并使用 metadata 中的 source rank slot mapping scatter 恢复 full visible compressed KV cache。
- chunk prefill 中，如果 C128/C4 compressor 输入需要当前 chunk 之前的 hidden token，`_assemble_dsa_cp_compressor_hidden_input()` 会从 `_dsa_cp_hidden_tail_cache[layer_name]` 取上一 chunk 的 per-request tail hidden；对应 prefix hidden 只参与 compressor 输入，不产生 cache 写入。
- 每层 local-cache prefill 结束后，`_save_dsa_cp_hidden_tail_cache()` 保存每个 request 最后最多 128 个有效 hidden token，作为下一 chunk 同一层的 prefix hidden 来源。

落点：

- `dsa_cp.py:585-604`: compressed positions 和 slot mapping size 生成。
- `dsa_cp.py:1191-1217`: compressor 调用和 scatter。
- `model_runner_v1.py:2985-3048`: 如必须由 runner 传入 local compressed positions/slot mapping，则在这里接入。
- `block_table.py:174-202`: 如选择在 block table 层生成 compressed local slot mapping，则扩展这里。

验收：

- C128 cache 与 full hidden 版本一致。
- all-gather 后 compressed token 顺序一致。
- pad token 不写 cache。
- local compressor scatter 使用 `compress_local_slot_mapping`，不会写到 full compressed token 流前缀。
- compressor state cache 只写 broadcast 后 selected state blocks。

### Phase 4: C4/indexer local 化

目标：

- `_update_indexer_cache()` 支持 local hidden + overlap。
- indexer quant/full/scale cache 写入 local owner。
- `_indexer_select_topk()` 能读取完整逻辑 indexer cache。
- indexer state cache 通过 per-request owner broadcast 同步。

当前实现状态：

- `_update_indexer_cache()` 已从 `indexer_kv_scale_metadata.req_metadata.cp_metadata.compressor_slot_plan` 获取 local compressor plan。
- 开关开启且 local plan 存在时，indexer compressor 输入切到 local/overlap hidden，`cu_seqlens` 使用 `input_query_start_loc`，`state_block_table` 按 `request_indices` 裁剪，`start_pos` 使用 `start_pos + start_pos_offsets`。
- C4 indexer local RoPE 已由 `compressed_positions` 直接生成，与 local compressor output 顺序一致。
- indexer cache scatter 前使用 `valid_output_mask` 过滤 borrowed output，只写 owner output；避免把 `-1` slot mapping 传给 indexer scatter。
- `_indexer_select_topk()` 仍按当前 cache 视图读取；当前已在调用 topk 前通过 owner update allgather 恢复本轮 full visible indexer cache。
- indexer state cache per-request owner 复用 `DSACPStateBroadcastPlan`；selected state block broadcast 已在 `_update_indexer_cache()` 尾部接入。
- chunk prefill 的 C4 indexer prefix hidden 复用 main compressor 的 hidden tail cache 机制；前序 4 token 缺失或不连续时直接报错，避免用错误值更新 indexer cache。

落点：

- `dsa_cp.py:1171-1189`
- `dsa_cp.py:1312-1368`

验收：

- C4A 前序 4 token 边界正确。
- topk 结果与 full hidden 版本一致。
- compressor overlap 开关开启/关闭都正确。
- 多 batch 下每个 request 的 state owner 独立正确。

### Phase 5: 性能和外部 cache 验证

目标：

- 去掉不必要的 full hidden all-gather。
- 保留必要的 halo/overlap、SWA/C4A/C128A cache allgather，以及 tail owner state broadcast。
- 验证 prefix cache、MooncakeStore load/save 不回退。

验收：

- CP64 prefill 峰值显存下降。
- compressor/indexer 重算量下降。
- 端到端输出一致。
- external cache 命中后 suffix 重算仍正确。

## 9. 风险

- `compressor` op 的 `cu_seqlens/start_pos/state_block_table` 是否允许 local segment 语义，需要小规模 NPU 验证。
- 如果 all-gather 后只恢复 compressed KV，但缺少 selected state broadcast，后续 op 可能读取不到每个 request 的最终 state。
- `slot_mapping` 和 `state_block_table` 一旦仍是 full 语义，local compressor 可能写错 compressed block。
- prefix cache hit 后的 suffix local segment 起点不是 0，halo/overlap 需要从逻辑位置而不是 batch offset 推导。
- metadata 构造尽量使用 CPU tensor/NumPy，避免在 hot path 增加 NPU `item()` 同步。

## 10. 第一版推荐收敛策略

第一版不要直接做“完全分布式 cache owner”。建议按以下保守边界实现：

```text
Q/local attention: local rank 计算
SWA/window: local hidden + 128 halo，local valid 写入
C128/C4/indexer: local hidden + overlap 计算
compressed/indexer 输出: all-gather 后恢复 full visible cache
compressor/indexer state: 从每个 request 最后 owner rank broadcast selected state blocks
prefix/MooncakeStore: 语义不变
```

这样可以先拿到主要收益：避免每个 rank 重复跑全量 hidden 的 compressor/indexer/SWA 前处理，同时把 external cache、scheduler、block hash 语义的改动推迟到第二版。


## 11. 当前代码落地状态补充

当前分支已把 local-cache CP 从过渡 full-hidden 路径推进到以下实现状态：

- 开关仍为 `VLLM_ASCEND_ENABLE_DSA_CP_LOCAL_CACHE`，默认关闭；关闭时继续走旧 DSA CP full hidden cache 路径。
- 开关开启且 prefill metadata 中存在 `DSACPLocalCachePlan` 时，`_forward()` 不再立即调用 `maybe_all_gather_and_maybe_unpad()` 获取 full hidden。
- 每层 forward 开始时调用 hidden halo allgather：每个 CP rank 发送自己当前 local slice 尾部最多 `128` 个 hidden token。接收端用 allgather 得到的 rank-tail hidden 和本 rank `hidden_states_local` 按 `DSACPHiddenInputPlan` 组装 SWA/compressor/indexer 需要的输入。
- `DSACPHiddenInputPlan` 记录 flattened token 输入范围，并区分：
  - `local_read_ranges` / `local_output_ranges`：从当前 rank `hidden_states_local` 读取。
  - `halo_source_ranges` / `halo_output_ranges`：从其它 rank 广播的 128-token tail hidden 读取。
- SWA/window cache 当前只计算当前 rank owner 的 valid token，并通过 `_all_gather_dsa_cp_swa_cache_updates()` allgather 本 rank owner SWA KV；slot mapping 不随层通信，而是使用 `DSACPSWAWindowPlan.all_rank_slot_mappings`。其它 rank scatter 后恢复本轮 full visible SWA cache。SWA 本身不使用 `swa_hidden_input_plan` 重新计算 halo token 的 KV；前序 128 token 应已由历史/prefix cache 或本轮其它 rank owner broadcast 写入 cache。
- C128/C4 main compressor 使用 `compressor_hidden_input_plan` 组装 local + overlap hidden，op 输出后用 `valid_output_mask` 过滤 overlap/borrowed output，只 scatter owner output，并通过 `_all_gather_dsa_cp_compressed_cache_updates()` allgather owner KV 恢复 full visible compressed KV cache。
- C4 indexer compressor 已接入同一套 `compressor_hidden_input_plan`，前序 4-token overlap 只参与计算，不写 cache；indexer k/full/scale cache 通过 `_all_gather_dsa_cp_indexer_cache_updates()` allgather owner KV 恢复 full visible 视图。
- `DSACPCompressorSlotPlan` 已记录 `prefix_lengths` 和 `current_start_positions`。当 chunk prefill 的 compressor 输入需要当前 chunk 之前的 hidden token 时，`_assemble_dsa_cp_compressor_hidden_input()` 会从 `_dsa_cp_hidden_tail_cache[layer_name]` 取前一 chunk 的 per-request tail hidden，并拼到当前 local/halo hidden 前面；tail 的结束 position 必须等于当前 chunk 起始 position，否则直接报错。
- 每层 local-cache prefill 结束后，`_save_dsa_cp_hidden_tail_cache()` 会按 request 保存最后 128 个有效 token 的 hidden state，作为下一 chunk 同一 layer 的 C4/C128 prefix hidden 来源。保存时同样通过 `build_dsa_cp_hidden_input_plan()` 从本 rank local hidden 和 rank-tail halo 中组装，padding token 不进入 tail cache。
- main compressor state cache 和 C4 indexer state cache 继续使用 `DSACPStateBroadcastPlan` 做 selected state block broadcast，不广播整块 state cache。
- SWA 的 per-rank allgather count/slot 由 `DSACPSWAWindowPlan.all_rank_valid_token_counts` 和 `all_rank_slot_mappings` 提供；compressor/indexer 的 per-rank allgather count/slot 由 `DSACPCompressorSlotPlan.all_rank_valid_output_counts` 和 `all_rank_slot_mappings` 提供。
- 变长 128-unit CP 切分下，`_restore_tp_head_layout()` 已在 local-cache 路径使用显式 `input_split_sizes/output_split_sizes` 调用 `dist.all_to_all_single()`，其中 output split 来自 `get_dsa_cp_all_rank_token_counts()`，可支持 CP64 9K 请求下 rank0-rank7 为 256 tokens、rank8-rank63 为 128 tokens 的不等长恢复。开关关闭时仍走旧的等长 `all_to_all_single()`。

仍需 NPU 实机验证的点：

- `torch.ops._C_ascend.compressor` 在 local `cu_seqlens/start_pos/state_block_table` 语义下的数值需要和 full-hidden 路径对齐验证。
- chunk prefill tail cache 当前优先按 `AscendCommonAttentionMetadata.request_ids` 传下来的稳定 request id 保存；只有 request id 不可用时才回退到 `idx:<req_idx>`。同时仍用 tail end position 校验连续性，避免相邻 chunk 不连续时复用错误 hidden。
- 本环境缺少 `pytest` 和 `torch` 运行依赖，当前只能完成 `py_compile` 级别验证；metadata UT 和 NPU 数值一致性需要在完整环境补跑。

## 12. 完整执行流程

### 12.1 总体阶段

```
┌─────────────────────────────────────────────────────────────┐
│                  Metadata 构建 (CPU, 每轮调度一次)            │
│  AscendDSACPMetadataBuilder.build()                         │
│  输入: scheduled requests, block_table, slot_mapping, ...    │
│  输出: DSACPMetadata (含各种 plan)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Forward 计算 (NPU, 每层执行一次)                 │
│  AscendDSACPImpl.forward() → _forward()                     │
│  输入: hidden_states (已 TP 切分的 local shard)              │
│  输出: attention output                                     │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 Metadata 构建阶段

```
build()
  │
  ├─ 判定开关: enable_dsa_cp_local_cache = ascend_envs.VLLM_ASCEND_ENABLE_DSA_CP_LOCAL_CACHE
  │
  ├─ 1. _build_local_cache_plan(num_input_tokens, query_start_loc) → DSACPLocalCachePlan | None
  │      └─ build_dsa_cp_local_cache_plan()
  │           └─ 每个 request 先 128 对齐，再在 flatten padded unit 流上分配连续 owner
  │           例: CP64, 9K=9216t → 72 units
  │               rank0-7:  2 units = 256 tokens
  │               rank8-63: 1 unit  = 128 tokens
  │               padded_total = 72*128 = 9216 (恰对齐，无 padding)
  │     条件: 开关开启 && cp_size > 1，否则返回 None
  │
  ├─ 2. _build_local_token_metadata()
  │      [老路径] local_cache_plan is None:
  │        flatten 均分, tokens_per_rank = ceil(num_tokens/cp_size)
  │        所有 rank 等长 (padding 补齐)
  │      [新路径] local_cache_plan is not None:
  │        使用 local_cache_plan.local_valid_ranges/local_offsets
  │        允许 rank 间不等长
  │
  ├─ 3. 构建 SWA cache plan:
  │      _build_swa_local_slot_mapping(local_cache_plan, num_actual_tokens)
  │        → swa_slot_mapping = slot_mapping[valid_start:valid_end]
  │        → swa_valid_start, swa_valid_end
  │
  │      _build_swa_window_plan(local_cache_plan, query_start_loc, num_actual_tokens)
  │        → DSACPSWAWindowPlan:
  │            input_ranges:       [(halo_start, valid_end)]
  │            valid_ranges:       [(valid_start, valid_end)]
  │            halo_ranges:        [(halo_start, valid_start)]
  │            all_rank_valid_token_counts: (rank0_count, rank1_count, ...)
  │            all_rank_slot_mappings:      (rank0_slots, rank1_slots, ...)
  │        halo 裁剪: 不能跨 request 借 token, 请求开头不足 128 token 时按实际裁剪
  │
  │      build_dsa_cp_hidden_input_plan(input_ranges, local_cache_plan) → swa_hidden_input_plan
  │        区分: local_read_ranges / local_output_ranges
  │              halo_source_ranges / halo_output_ranges
  │
  ├─ 4. 构建 compressor slot plan:
  │      _build_compressed_local_slot_mapping() → compressed_slot_mapping, compressed_valid_start/end
  │        └─ build_dsa_cp_local_compressed_range()
  │             local compressed output 在全局 compressed 流中的 [start, end)
  │
  │      _build_local_compressor_slot_plan() → DSACPCompressorSlotPlan:
  │        └─ build_dsa_cp_local_compressor_slot_plan()
  │             每 request:
  │               input_ranges:     [(input_start, valid_end)]  含 overlap + 可能的 prefix hidden
  │               valid_ranges:     [(valid_start, valid_end)]  当前 rank owner
  │               overlap_ranges:   [(input_start, valid_start)]
  │               prefix_lengths:   [N]  当前 chunk 之前需要的 prefix hidden token 数
  │               current_start_positions: [pos]  当前 chunk 起始绝对位置
  │               has_prefix_hidden: True/False
  │             全局:
  │               slot_mapping:     local compressor output 对应的 cache slot
  │                                 overlap/borrowed output → -1 (不写 cache)
  │               valid_output_mask: owner output → True, borrowed → False
  │               compressed_positions:  local output 对应的 RoPE position
  │               input_indices:    local input 在 full hidden 中的下标
  │               input_query_start_loc:  local per-request cu_seqlens
  │               request_indices:  local input 对应的原始 request 行
  │               start_pos_offsets: 每段 input 在 request 内的起始偏移
  │               all_rank_valid_output_counts: (rank0_count, ...)
  │               all_rank_slot_mappings:      (rank0_slots, ...)
  │
  │      build_dsa_cp_hidden_input_plan(input_ranges, ...) → compressor_hidden_input_plan
  │
  ├─ 5. _build_state_broadcast_plan() → DSACPStateBroadcastPlan:
  │      └─ build_dsa_cp_state_broadcast_plan()
  │          每 request:
  │            找到 tail token 的 CP owner rank
  │            计算 state_block_index = tail_position // compress_ratio // state_block_size
  │            从 state_block_table[req_idx, state_block_index] 取 state_block_id
  │            source_ranks[req_idx] = tail_owner_rank (或 -1 表示空请求)
  │            local_request_indices: 当前 rank 需要作为 broadcast 源的 request 集合
  │            state_block_ids[req_idx]: tail state 对应的 cache block id
  │            state_valid_mask[req_idx]: 本轮是否需要同步
  │
  ├─ 6. compressed RoPE:
  │      [新路径] compressor_slot_plan is not None:
  │        compressed_positions = compressor_slot_plan.compressed_positions
  │        compress_cos, compress_sin = get_cos_and_sin_dsa(compressed_positions, use_cache=False)
  │      [老路径]:
  │        compressed_positions = _get_padded_compressed_position(...)
  │        compress_cos, compress_sin = get_cos_and_sin_dsa(..., use_cache=not has_prefill)
  │
  └─ 7. 组装 DSACPMetadata:
       local_cache_plan, swa_window_plan, swa_hidden_input_plan,
       swa_slot_mapping, swa_valid_start/end,
       compressor_slot_plan, compressor_hidden_input_plan,
       compressed_slot_mapping, compressed_valid_start/end,
       state_broadcast_plan
```

### 12.3 Forward 计算阶段

#### 12.3.1 入口 `forward()`

```text
forward(layer_name, hidden_states, kv_cache, attn_metadata, output)
  │  hidden_states: 本层输入，已按 TP 切分到本 rank 的 local shard
  │
  ├─ _forward(layer_name, hidden_states, kv_cache, attn_metadata)
  │   返回: local_attn_output  shape: [local_num_tokens, n_local_heads, head_dim]
  │
  ├─ _restore_tp_head_layout(local_attn_output, layer_name, attn_metadata)
  │   │ 1. reverse RoPE (local_cos, -local_sin)
  │   │ 2. all_to_all_single 恢复完整 token 顺序
  │   │    [老路径] local_cache_plan is None:
  │   │      input_split = output_split = [N]*tp_size  (等长)
  │   │    [新路径] local_cache_plan is not None:
  │   │      output_split = get_dsa_cp_all_rank_token_counts(local_cache_plan)
  │   │      支持 rank 间不等长恢复 (如 256,256,...,128,128)
  │   └─ → o_proj_input  shape: [total_tokens, n_local_heads, head_dim]
  │
  └─ wo_a → wo_b → output  (输出投影)
```

#### 12.3.2 核心 `_forward()` 详细流程

```text
_forward(layer_name, hidden_states_local, kv_cache, attn_metadata)
  │
  ├─ 解包 kv_cache:
  │   compress_kv_cache, swa_kv_cache, state_cache = unpack(kv_cache)
  │
  ├─ 解包 attn_metadata (按 compress_ratio):
  │   compress_ratio==4:  compressor_attn, compressor_kv_state, _, _, swa
  │   compress_ratio==128: compressor_attn, compressor_kv_state, swa
  │   compress_ratio<=1:   swa
  │
  ├─ 判定路径:
  │   has_prefill = _has_prefill(attn_state)
  │   use_local_cache_prefill = has_prefill && cp_metadata.local_cache_plan is not None
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 0: Hidden 准备
  │  ═══════════════════════════════════════════════════════════════
  │
  │  [新路径] use_local_cache_prefill=True:
  │    hidden_halos = _gather_dsa_cp_hidden_halos(hidden_states_local, local_cache_plan, num_actual_tokens)
  │      → 每个 rank 准备自己 local slice 尾部最多 128 token 的 hidden
  │      → dist.all_gather 收集所有 rank 的 tail hidden
  │      → 返回 (halo_buffers: list[Tensor], halo_ranges: list[(start,end)])
  │    hidden_states_cache = hidden_states_local  直接用 local
  │
  │  [老路径] use_local_cache_prefill=False:
  │    hidden_states = maybe_all_gather_and_maybe_unpad(hidden_states_local)  all-gather full hidden
  │    hidden_states_cache = hidden_states[:num_actual_tokens]
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 1: Q 计算 (两路径相同，始终用 local hidden)
  │  ═══════════════════════════════════════════════════════════════
  │
  │  qr = q_norm(wq_a(hidden_states_local))
  │  q  = wq_b(qr)
  │  q  = unflatten → apply_dsa_q_rms → RoPE(local_cos, local_sin)
  │  Q shape: [local_num_tokens, num_heads, head_dim]
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 2: SWA KV Cache
  │  ═══════════════════════════════════════════════════════════════
  │
  │  [新路径]:
  │    swa_hidden = hidden_states_local[local_read_start:local_read_end]  只取 valid owner token
  │    swa_cos/sin = cos/sin[valid_start:valid_end]                       只取 valid RoPE
  │    swa_slot_mapping = cp_metadata.swa_slot_mapping                     owner 对应的 cache slot
  │    swa_kv = wkv(swa_hidden) → kv_norm → RoPE
  │    dsa_kv_compress_scatter(swa_kv_cache, swa_kv, swa_slot_mapping)   写入 owner cache
  │
  │    if swa_window_plan is not None:
  │      _all_gather_dsa_cp_swa_cache_updates(swa_kv_cache, swa_kv, window_plan)
  │        → allgather 各 rank 的 owner SWA KV
  │        → 用 metadata 中预计算的 all_rank_slot_mappings scatter 到其它 rank
  │        → 恢复 full visible SWA cache
  │
  │  [老路径]:
  │    swa_kv = wkv(hidden_states_cache)  ← full hidden
  │    kv_norm → RoPE(full cos/sin) → dsa_kv_compress_scatter(full slot_mapping)
  │    (每个 rank 独立算 full hidden，无需 allgather 同步)
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 3: Compressor (C128/C4) — 仅 compress_ratio > 1
  │  ═══════════════════════════════════════════════════════════════
  │
  │  [新路径] compressor_cp_metadata.compressor_slot_plan is not None:
  │
  │    3a. 组装 compressor 输入 hidden:
  │        compressor_hidden_states = _assemble_dsa_cp_compressor_hidden_input(
  │            layer_name, slot_plan, hidden_input_plan,
  │            hidden_states_local, hidden_halos, hidden_states_full)
  │          │
  │          ├─ _assemble_dsa_cp_hidden_input(hidden_input_plan, local, halos)
  │          │    从 local hidden 和 halo buffers 按 output_ranges 拼接当前 chunk 所需 hidden
  │          │    → current_hidden  shape: [input_len, ...]
  │          │
  │          └─ if slot_plan.has_prefix_hidden:
  │               从 _dsa_cp_hidden_tail_cache[layer_name][req_key] 取 prefix hidden
  │               校验: tail_end_pos == current_start_pos (连续性检查)
  │               将 prefix hidden 拼到 current_hidden 前面
  │             → full_input = [prefix | current_hidden]
  │
  │    3b. 替换 compressor 参数为 local 语义:
  │        compressor_cu_seqlens    = slot_plan.input_query_start_loc   local cu_seqlens
  │        compressor_start_pos     = start_pos[request_indices] + start_pos_offsets
  │        compressor_state_block_table = state_block_table[request_indices]
  │        compressor_sin/cos       = 由 slot_plan.compressed_positions 生成
  │
  │    3c. compressed_kv = compressor_op(compressor_hidden_states, ...)
  │
  │    3d. 过滤 borrowed output:
  │        valid_mask = slot_plan.valid_output_mask.to(device)
  │        compressed_kv = compressed_kv[valid_mask]           过滤掉 overlap/borrowed output
  │        compressor_slot_mapping = slot_mapping[valid_mask]  只保留 owner slot
  │
  │    3e. if compressed_kv is not None:
  │          dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compressor_slot_mapping)
  │
  │    3f. if compressor_slot_plan is not None:
  │          _all_gather_dsa_cp_compressed_cache_updates(compress_kv_cache, compressed_kv, slot_plan)
  │            → allgather 各 rank 的 owner compressed KV
  │            → 用 metadata all_rank_slot_mappings scatter 恢复 full visible compressed cache
  │
  │    3g. _broadcast_dsa_cp_state_blocks(state_cache, compressor_state_broadcast_plan)
  │          每 request: tail owner rank broadcast 自己的 selected state block
  │          其他 rank: state_cache[block_id].copy_(received_block)
  │
  │  [老路径] compressor_slot_plan is None:
  │    compressed_kv = compressor_op(
  │        hidden_states_cache,              ← full hidden
  │        full cu_seqlens, full start_pos, full state_block_table,
  │        full compress_cos/sin, ...)
  │    dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, full_slot_mapping)
  │    (无 allgather, 无 state broadcast)
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 3a: C4 Indexer Cache — 仅 compress_ratio==4
  │  ═══════════════════════════════════════════════════════════════
  │
  │  _update_indexer_cache()  与 compressor 完全对称:
  │
  │  [新路径] indexer_slot_plan is not None:
  │    1. 组装: _assemble_dsa_cp_compressor_hidden_input() → indexer_x
  │       (复用同一套 compressor_slot_plan 和 hidden_input_plan)
  │    2. kv = compressor_op(indexer_x, local_params, ...)
  │    3. 过滤: kv = kv[valid_output_mask]
  │    4. scatter: rotate → indexer_quant_scatter_part1 → scatter_scale_part3
  │       (使用过滤后的 local slot_mapping)
  │    5. _all_gather_dsa_cp_indexer_cache_updates(indexer_k, scale, full, kv, slot_plan)
  │    6. _broadcast_dsa_cp_state_blocks(indexer_state_cache, state_broadcast_plan)
  │
  │  [老路径] indexer_slot_plan is None:
  │    kv = compressor_op(full_hidden, ...)
  │    rotate → quant_scatter → scale_scatter
  │    (无 allgather, 无 state broadcast)
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 3b: Indexer Select TopK (C4 稀疏查询)
  │  ═══════════════════════════════════════════════════════════════
  │
  │  _indexer_select_topk(hidden_states_local, qr, kv_cache, ...)
  │    1. indexer Q = inderxer_wq_b(qr) → RoPE → rotate_activation
  │    2. weights = weights_proj(hidden_states_local)
  │    3. quantize query
  │    4. topk_idxs = npu_quant_lightning_indexer(
  │         q, indexer_k_cache, weights, indexer_scale_cache, ...)
  │    → topk_idxs 用于稀疏注意力选择
  │    注意: 查询用 local Q, indexer_k_cache 在 allgather 后已是 full visible
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 4: Sparse Attention
  │  ═══════════════════════════════════════════════════════════════
  │
  │  attn_output = attn_op(
  │      q,
  │      ori_kv=swa_kv_cache,            SWA 窗口 KV cache (full visible)
  │      cmp_kv=compress_kv_cache,       压缩 KV cache (full visible)
  │      cmp_sparse_indices=topk_idxs,   C4 indexer 选出的 topk
  │      ori_block_table=..., cmp_block_table=...,
  │      cu_seqlens_q=local_seq_lengths_query,
  │      seqused_kv=local_seq_lengths_key,
  │      metadata=sas_metadata, ...)
  │  → attn_output  shape: [local_num_tokens, n_local_heads, head_dim]
  │
  ├─ ═══════════════════════════════════════════════════════════════
  │  步骤 5: 保存 Hidden Tail Cache (仅新路径)
  │  ═══════════════════════════════════════════════════════════════
  │
  │  if use_local_cache_prefill:
  │    _save_dsa_cp_hidden_tail_cache(layer_name, hidden_states_local, hidden_halos,
  │                                    req_metadata, local_cache_plan, num_actual_tokens)
  │      1. 计算每个 request 的 tail 范围: [req_end-128, req_end)
  │      2. build_dsa_cp_hidden_input_plan(tail_ranges, ...) 组装 tail plan
  │      3. _assemble_dsa_cp_hidden_input() 从 local+halo 拼装 tail hidden
  │      4. 存入 _dsa_cp_hidden_tail_cache[layer_name] = {
  │           req_key: (tail_end_position, tail_hidden_tensor)
  │         }
  │      5. 下一 chunk 同一层的步骤 3a 从此缓存读取 prefix hidden
  │      6. padding token 不进入 tail cache (通过 num_actual_tokens 裁剪)
  │
  └─ return attn_output
```

### 12.4 每层通信汇总（新路径）

| 序号 | 通信操作 | 时机 | 发送内容 | 方式 | 同步目标 |
|------|----------|------|----------|------|----------|
| 1 | Hidden Halo Gather | 每层开始时 | 各 rank local tail 最多 128 hidden token | `all_gather` | 其他 rank 组装 halo input |
| 2 | SWA Cache Sync | SWA scatter 后 | 本 rank owner SWA KV update | `all_gather` + metadata slot | 恢复 full visible SWA cache |
| 3 | Compressed Cache Sync | compressor scatter 后 | 本 rank owner compressed KV update | `all_gather` + metadata slot | 恢复 full visible compressed cache |
| 4 | Indexer Cache Sync | indexer scatter 后 | 本 rank owner indexer KV update | `all_gather` + metadata slot | 恢复 full visible indexer cache |
| 5 | Compressor State Broadcast | compressor sync 后 | tail owner 的 selected state block | per-request `broadcast` | 各 rank state cache 对应 block |
| 6 | Indexer State Broadcast | indexer sync 后 | tail owner 的 selected state block | per-request `broadcast` | 各 rank state cache 对应 block |
| 7 | Head Layout Restore | `forward()` 返回前 | attention output | `all_to_all_single` (变长) | 恢复完整 token 顺序 |

### 12.5 新旧路径分叉点

```text
forward() 入口
  │
  _forward()
  ├─ has_prefill && cp_metadata.local_cache_plan is not None ?
  │   │
  │   ├─ YES → 新 local-cache CP 路径
  │   │   ├─ hidden halo allgather (rank tail 128 hidden)
  │   │   ├─ SWA: local valid only → scatter → allgather sync
  │   │   ├─ compressor: local+overlap+prefix → op → filter → scatter → allgather sync
  │   │   ├─ indexer: local+overlap+prefix → op → filter → scatter → allgather sync
  │   │   ├─ state: per-request selected block broadcast (tail owner→others)
  │   │   └─ save tail hidden cache (供下一 chunk 同层使用)
  │   │
  │   └─ NO  → 老 full-hidden CP 路径 (默认)
  │       ├─ maybe_all_gather_and_maybe_unpad() → full hidden
  │       ├─ SWA: full hidden → full scatter
  │       ├─ compressor: full hidden → full scatter
  │       ├─ indexer: full hidden → full scatter
  │       └─ 无 allgather sync, 无 state broadcast
  │
  _restore_tp_head_layout()
  ├─ cp_metadata.local_cache_plan is not None ?
  │   ├─ YES → 变长 all_to_all_single (output_split 按 128-unit 分配)
  │   └─ NO  → 等长 all_to_all_single
  │
  wo_a → wo_b → output
```

### 12.6 Hidden State 跨 Chunk 传递（Chunk Prefill）

```text
Chunk 1, Layer L:
  _forward() 入口: hidden_states_local = layer L 的输入 (来自 layer L-1 的输出)
  ...
  步骤 3: compressor 需要 prefix hidden?
          → 这是第一个 chunk，has_prefix_hidden=False，跳过
  ...
  步骤 5: _save_dsa_cp_hidden_tail_cache(layer_name="model.layers.L.self_attn", ...)
          → 保存每个 request tail 128 hidden 到 _dsa_cp_hidden_tail_cache[layer_name]

Chunk 2, Layer L (同一层，不同 chunk):
  _forward() 入口: hidden_states_local = layer L 的新输入 (layer L-1 处理 chunk2 后的输出)
  ...
  步骤 3: compressor 需要 prefix hidden?
          → has_prefix_hidden=True
          → _assemble_dsa_cp_compressor_hidden_input()
            → 从 _dsa_cp_hidden_tail_cache["model.layers.L.self_attn"][req_key]
              取出 chunk1 保存的 tail hidden
            → 校验 tail_end_pos == current_start_pos
            → 拼接到 compressor 输入前面
            → prefix hidden 只参与计算，不产生 cache 写入
  ...
  步骤 5: _save_dsa_cp_hidden_tail_cache(...)  ← 覆盖为 chunk2 的 tail

关键约束:
  - tail cache key = layer_name，保证同一层跨 chunk 传递
  - req_key 优先使用 request_ids (稳定 id)，回退 idx:<req_idx>
  - 保存时通过 num_actual_tokens 裁剪，padding token 不进缓存
  - 读取时校验 tail_end_pos == current_start_pos，不连续则报错
```

### 12.7 新 CP padding 策略

当前方案采用 per-request 128-token logical padding + flattened ownership：

```text
for each request:
  real_len = query_start_loc[i + 1] - query_start_loc[i]
  padded_units = ceil(real_len / 128)
flatten padded units across requests
owner range = rank 在 flattened padded unit 流中分到的连续 128-token unit
valid range = owner range 映射回真实 flattened token 后的有效 token，padding token 不进入 attention / cache write
```

关键约束：

1. `query_start_loc`、`seq_lens`、KV len 仍表示真实 token 长度，不包含 logical padding。
2. `DSACPLocalCachePlan.rank_valid_ranges` 保留每个 rank、每个 request 的真实 owner token。
3. `local_valid_ranges + local_offsets` 描述本 rank compact local hidden 的布局；多 batch 下它不再等价于一个连续的 `[local_start, local_end)`。
4. SWA halo 按 request 裁剪，不能从前一个 request 借 token。
5. C4/C128/state 的跨 rank cache 状态仍通过 all-gather/broadcast 恢复为全局可见。

## 13. Prefix Cache 子计划

### 13.1 阅读结论

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

### 13.2 两条容易混淆的路径

#### 13.2.1 当前工程的 `enable_dsa_cp`

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

#### 13.2.2 后续迁移 Geneva CP 后的路径

如果未来把 `/home/xujiuxu/code/omniinfergeneva2` 仓的 DeepSeekV4 CP 方案迁移进来，语义会多一个变化：wincache 可能变成 CP rank 局部 ownership。

但需要注意：

```text
CP rank 局部拥有 wincache block
  != MooncakeStore lookup 要按 CP rank 匹配
```

只要 save 到 MooncakeStore 的 value 是完整 cache block，lookup 仍然按逻辑 block hash 匹配，不需要把 CP rank 放进 key。

### 13.3 DeepSeekV4 多 Cache Group

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

### 13.4 不接 MooncakeStore 时的 Prefix Cache

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

### 13.5 接 MooncakeStore 时的 Prefix Cache

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

### 13.6 MooncakeStore Key 语义

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

### 13.7 Lookup: 为什么用 C1 做 Gate

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

### 13.8 Load: 命中后会尝试加载哪些 Cache

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

### 13.9 Save: 计算完成后如何写 MooncakeStore

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

### 13.10 DSA CP 与 TP 的边界

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

### 13.11 当前实现下的 FAQ

#### 13.11.1 开启 `enable_dsa_cp` 会影响 PCP/DCP 吗

不会。

`enable_dsa_cp=True` 只选择 DeepSeekV4 DSA attention 的内部 CP 执行路径。`PCP=1、DCP=1` 时，prefix cache 仍按非 PCP/DCP 的本地或外部 KV pool prefix cache 理解。

#### 13.11.2 C1 命中是否可能比 C4/C128 更多

可能。

当前 MooncakeStore lookup 使用 C1 dense group 做 gate。如果 C4 或 C128 某些 key 缺失，load 阶段会记录 invalid blocks，scheduler 会回退并重算缺失 suffix。

#### 13.11.3 MooncakeStore 是否用一个 hash 匹配所有 DeepSeekV4 cache

不是。

相同 chunk hash 会被包装成带 `group` 和 `cache_family` 的完整 key。C1、C4、C128 是不同 key space。

#### 13.11.4 TP rank 上传的 key 是否一定代表 TP 语义分片

不一定。

当前 save path 可以按 TP rank 分摊上传 key，但这更像上传任务拆分。对于 replicated attention cache，不能简单理解为“每个 TP rank 只拥有自己上传的那部分逻辑 cache”。

### 13.12 引入 Geneva CP 后的 MooncakeStore 适配方案

本节讨论另一个目标方案：如果把 `omniinfergeneva2` 仓的 DeepSeekV4 CP 方案迁移到当前工程，并采用“方案 2”的 MooncakeStore key 语义。

这里的核心假设是：DeepSeekV4 attention cache 是 replicated cache，不是 TP/head 维度语义分片 cache。

#### 13.12.1 与 PD `mooncake_hybrid_connector` 分开看

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

#### 13.12.2 方案 2: replicated attention cache 使用统一 logical TP key

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

#### 13.12.3 Geneva CP 后 Save 怎么做

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

#### 13.12.4 Geneva CP 后 Lookup 怎么做

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

#### 13.12.5 Geneva CP 后 Load 怎么做

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

#### 13.12.6 Geneva CP 后多 Cache Group 怎么处理

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

#### 13.12.7 推荐修改点

如果按方案 2 落地，MooncakeStore 相关代码建议按以下方向修改。

##### 13.12.7.1 统一 replicated attention cache key

对 DeepSeekV4 replicated attention cache，生成 `PoolKey` 时使用：

```text
head_or_tp_rank = 0
```

或明确 sentinel：

```text
head_or_tp_rank = replicated
```

不能让真实 TP rank 进入 replicated attention cache 的 key identity。

##### 13.12.7.2 Save 阶段支持 CP block ownership

保存 wincache 时，key list 按当前 CP rank 拥有的完整 blocks 过滤：

```text
all computed blocks
  -> filter blocks owned by current cp_rank
  -> put owned complete blocks
```

过滤必须基于 cache block 边界，不能基于 block 内 token 范围。

##### 13.12.7.3 Load 阶段按完整 cache 需求加载

load 侧不按 CP owner 过滤 key。每个需要完整 attention cache 的 rank 都构造完整命中 prefix 的 key list：

```text
matched prefix blocks
  -> all block keys
  -> m_store.get(all keys)
  -> fill local KV/cache blocks
```

##### 13.12.7.4 Lookup 阶段保持逻辑 block hash 匹配

lookup 阶段不引入 CP rank 条件：

```text
block_hash + group/cache_family + logical replicated rank
```

这就是 cache identity。

##### 13.12.7.5 失败处理沿用 invalid block 回退

如果某些 CP rank 没有成功 put 自己拥有的 blocks，后续可能出现：

```text
C1 gate hit
load missing block
```

这时仍应沿用当前 invalid block 机制，让 scheduler 截断并重算缺失 suffix。

### 13.13 Geneva CP 方案复杂度

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

### 13.14 最终总结

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

### 13.15 MooncakeStore 新 CP 适配

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

### 13.16 新 CP 的 128-token 尾部与 16K Prefix-cache Put Owner

新 local-cache CP 的 128-token halo 只服务模型侧 SWA/window cache 计算；它不改变外部 prefix-cache 的传输粒度。外部 put 仍只在完整 cache-transfer granularity（DeepSeekV4 常见为 16K）达到时执行，未达到边界的尾部不会单独 put。

对于每个完整的 16K block，put owner 按该 block 最后一个 token（`boundary - 1`）所在的 CP rank 选择：

1. **128 token 完整落在一个 rank**：该 rank 拥有 `boundary - 1`，负责这个 16K block 的外部 put；其他 rank 不重复 put。各 rank 仍可写自己的本地 SWA/window cache 副本。
2. **128 token 跨 rank**：右侧、即拥有 `boundary - 1` 的 rank 负责完整 16K block 的 put。block 即使跨越 owner 范围，也不能因此丢弃；过滤逻辑按 block 末 token 归属选择唯一 owner。
3. **128 token 跨 chunk**：前一 chunk 已保存的 state/cache 直接复用，不新增全量 all-gather。达到 16K 边界的 chunk 仍由最终拥有 `boundary - 1` 的 rank put；未达到边界则等待后续 chunk。

这种 owner 过滤只作用于外部 SWA/prefix-cache block put，不影响老 CP，也不限制 batch 或 chunk 的总 token 数必须是 128 的倍数。新 CP 内部仍可按 128-token unit 做 halo 和 kernel 对齐。
