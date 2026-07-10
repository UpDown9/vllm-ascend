# MooncakeLayerwiseConnector 完整数据流（简化版）

## 系统架构

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ MooncakeLayerwiseConnector 架构                                                 │
└────────────────────────────────────────────────────────────────────────────────┘

    P节点 (Prefiller)       D节点 (Decoder)        Proxy
    ├─8个发送线程             ├─8个接收线程           ├─FastAPI + Uvicorn
    ├─ThreadPool(32线程)      ├─ThreadPool(32线程)    ├─asyncio协程池
    ├─KVCacheSendingLayer     ├─KVCacheRecvingLayer   ├─httpx.AsyncClient
    │  Thread                 │  Thread               │
    ├─Mooncake TransferEngine ├─Mooncake TransferEngine├─/v1/metaserver端点
    ├─is_kv_producer=True     ├─is_kv_consumer=True   ├─负载均衡逻辑
    └─监听send_queue          └─监听ZMQ端口7000-7007  └─req_data_dict缓存
```

## 完整数据流流程图

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ MooncakeLayerwiseConnector 完整数据流（简化版）                                 │
└────────────────────────────────────────────────────────────────────────────────┘

    P节点                    D节点                    Proxy
    │                        │                        │
    │                        │                        │ T1: Client请求到达
    │                        │                        │ ↓
    │                        │ ←──────────────────── HTTP POST（初始请求）
    │                        │ (prompt + kv_transfer_params)
    │                        │                        │
    │                        │ T2: Scheduler处理      │
    │                        │ ├─分配blocks [5,6,7]   │
    │                        │ ├─ThreadPool异步提交   │
    │                        │ ↓                      │
    │                        │ ─────────────────────→ HTTP POST（ThreadPool）
    │                        │ kv_transfer_params（D节点信息）
    │                        │                        │
    │                        │                        │ T3: metaserver处理
    │                        │                        │ ├─选择P节点
    │                        │                        │ ↓
    │ ←──────────────────────────────────────────── HTTP POST（转发）
    │ prompt + kv_transfer_params（D节点IP、端口）
    │                        │                        │
    │ T4: Scheduler处理      │                        │
    │ ├─分配blocks [10,11,12]│                        │
    │ ├─添加到send队列       │                        │
    │ ↓                      │                        │
    │                        │                        │
    │ T5: Layer0计算         │                        │
    │ ├─ZMQ查询metadata      │                        │
    │ ─────────────────────→ ZMQ REQ（GET_META_MSG） │
    │                        │                        │
    │                        │ ←────────────────────── ZMQ ROUTER（metadata）
    │ ←───────────────────── {te_rpc_port, layer_metadata}
    │                        │                        │
    │ ├─缓存metadata         │                        │
    │ ├─构建SendTask         │                        │
    │ ├─put send_queue       │                        │
    │ ↓                      │                        │
    │                        │                        │
    │ [后台线程] Layer0传输  │                        │
    │ ─────────────────────→ Mooncake RDMA（KV cache）
    │ (src=0x7fa0000 → dst=0x8fa0000)
    │                        │ 直接写入内存           │
    │                        │                        │
    │ T6: Layer1计算         │                        │
    │ ├─直接使用缓存         │                        │
    │ ├─put send_queue       │                        │
    │ ↓                      │                        │
    │                        │                        │
    │ [后台线程] Layer1传输  │                        │
    │ ─────────────────────→ Mooncake RDMA           │
    │                        │                        │
    │ ... Layer2-58重复 ...  │                        │
    │                        │                        │
    │ T61: Layer59计算       │                        │
    │ （最后层）             │                        │
    │ ├─chunk_finish=True    │                        │
    │ ├─put send_queue       │                        │
    │ ↓                      │                        │
    │                        │                        │
    │ [后台线程] Layer59传输 │                        │
    │ ─────────────────────→ Mooncake RDMA           │
    │                        │                        │
    │ 发送完成通知           │                        │
    │ ─────────────────────→ ZMQ REQ（DONE_SENDING_MSG）
    │ (8个rank并行发送)      │                        │
    │                        │                        │
    │                        │ 收到8个通知            │
    │                        │ ├─task_tracker记录     │
    │                        │ ├─done_requests.add    │
    │                        │ ←────────────────────── ZMQ ROUTER（ACK）
    │ ←───────────────────── （8次响应）              │
    │                        │                        │
    │ HTTP响应返回           │                        │
    │ ─────────────────────────────────────────────→ HTTP（prefill完成）
    │                        │                        │
    │                        │                        │
    │                        │ T63: Worker查询        │
    │                        │ get_finished()         │
    │                        │ ├─恢复请求             │
    │                        │ ↓                      │
    │                        │                        │
    │                        │ T64: Decode执行        │
    │                        │ ├─使用已加载KV cache   │
    │                        │ ├─生成token            │
    │                        │ ↓                      │
    │                        │                        │
    │                        │ ─────────────────────→ HTTP Stream（token流）
    │                        │                        │
    │                        │                        │ ─────────────────────→ Client
    │                        │                        │ （token流转发）
    │                        │                        │
    │                        │                        │
    ▼                        ▼                        ▼
  完成                    完成                      完成
```

## 通信统计

```
┌──────────────────┬──────────────┬──────────────────────────────────────────────┐
│ 通信类型          │ 通道          │ 次数                                          │
├──────────────────┼──────────────┼──────────────────────────────────────────────┤
│ Proxy → D        │ HTTP          │ 1次（初始请求）                                │
│ D → Proxy        │ HTTP（ThreadPool）│ 1次（异步提交kv_transfer_params）         │
│ Proxy → P        │ HTTP          │ 1次（metaserver转发）                          │
│ P → D            │ ZMQ REQ       │ 8次（Layer 0查询metadata，每个rank一次）      │
│ D → P            │ ZMQ ROUTER    │ 8次（返回metadata）                            │
│ P → D            │ Mooncake RDMA │ 480次（60层 × 8 ranks，每层每个rank传输一次） │
│ P → D            │ ZMQ REQ       │ 8次（最后层完成通知，每个rank一次）            │
│ D → P            │ ZMQ ROUTER    │ 8次（ACK响应）                                 │
│ P → Proxy        │ HTTP          │ 1次（prefill完成响应）                         │
│ D → Proxy        │ HTTP Stream   │ N次（token流，持续传输）                       │
│ Proxy → Client   │ HTTP Stream   │ N次（转发token流）                             │
├──────────────────┼──────────────┼──────────────────────────────────────────────┤
│ 总计              │               │ HTTP: 4次                                      │
│                  │               │ ZMQ: 16次（8查询+8通知）                        │
│                  │               │ Mooncake: 480次                                │
└──────────────────┴──────────────┴──────────────────────────────────────────────┘
```

## 时间轴关键事件

```
T0:   系统启动
      - P节点：启动8个发送线程，记录metadata
      - D节点：启动8个接收线程，监听ZMQ端口7000-7007
      - Proxy：启动FastAPI，监听端口9000

T1:   客户端请求到达Proxy
      - Proxy选择D节点，转发请求

T2:   D节点Scheduler处理
      - 标记WAITING_FOR_REMOTE_KVS
      - 分配blocks [5,6,7]
      - ThreadPool异步提交到Proxy

T3:   Proxy metaserver处理
      - 选择P节点
      - 转发请求（包含D节点连接信息）

T4:   P节点Scheduler处理
      - 分配blocks [10,11,12]
      - 添加到_reqs_need_send_layerwise队列

T5-T6:   P节点Layer 0计算+传输
      - Worker计算Layer 0
      - ZMQ查询D节点metadata（第一次）
      - 缓存metadata（所有层地址）
      - 后台线程传输Layer 0

T7-T60:  P节点Layer 1-58计算+传输
      - 每层：直接使用缓存metadata
      - 计算与传输并行
      - 无需ZMQ查询

T61-T62: P节点Layer 59（最后层）传输
      - 后台线程传输Layer 59
      - 发送DONE_SENDING_MSG（8个rank并行）
      - D节点收到8个通知，标记完成
      - P节点返回HTTP响应到Proxy

T63-T64: D节点恢复Decode
      - Worker查询get_finished()
      - Scheduler恢复请求
      - Worker执行decode
      - Stream token到Proxy → Client
```

## 关键数据结构传递链

```
Client → Proxy
  {prompt: "Hello", max_tokens: 16}

Proxy → D节点
  {prompt: ..., kv_transfer_params: {do_remote_prefill: True, metaserver: ...}}

D节点 → Proxy（ThreadPool）
  kv_transfer_params: {
    request_id: "abc123",
    remote_host: "D_ip",           ← D节点IP
    remote_port: 7000,             ← ZMQ查询端口
    remote_block_ids: [5,6,7],     ← D节点分配的blocks
    remote_engine_id: "dec_1",
    ...
  }

Proxy → P节点
  {
    prompt: ...,
    kv_transfer_params: {          ← D节点信息
      remote_host: "D_ip",
      remote_port: 7000,
      remote_block_ids: [5,6,7],
      do_remote_decode: True,
      max_completion_tokens: 1
    }
  }

P节点 → D节点（ZMQ查询）
  (GET_META_MSG, request_id)

D节点 → P节点（ZMQ响应）
  MooncakeAgentMetadata: {
    te_rpc_port: 6000,             ← Mooncake RPC端口
    layer_metadata: {
      "layer0": {base_addr: 0x8f0000, block_len: 16384},
      "layer1": {base_addr: 0x8f1000, ...},
      ...
    }                              ← 所有层的内存地址
  }

P节点 → D节点（Mooncake传输，每层）
  src: P节点内存地址 (0x7fa0000)
  dst: D节点内存地址 (0x8fa0000)    ← 使用metadata计算
  length: KV cache数据大小

P节点 → D节点（ZMQ完成通知）
  (DONE_SENDING_MSG, request_id, trans_count=8, side_channel_path)

D节点内部
  task_tracker[request_id]: {"P_ip:7000", ..., "P_ip:7007"}
  done_requests: {"abc123"}        ← 完成集合
```

## 计算与传输Overlap示意

```
P节点主线程：
Layer0计算  [====]
           ↓ put SendTask
Layer1计算      [====]
               ↓ put SendTask
Layer2计算           [====]
                    ↓ put SendTask
...

P节点后台线程：
Layer0传输      [====]  ← 从send_queue取
Layer1传输           [====]
Layer2传输                [====]
...

Overlap效果：
- Layer0传输时，Layer1正在计算
- 减少整体等待时间
- 充分利用NPU和RDMA并行能力
```

## 关键设计亮点

1. **ThreadPool异步提交**：D节点Scheduler不阻塞，异步提交metaserver请求
2. **Metadata缓存机制**：Layer 0查询后，后续59层直接使用缓存，节省约2950ms
3. **计算与传输Overlap**：后台线程传输时，主线程继续计算下一层
4. **每个TP Rank独立线程**：8个rank并行传输，充分利用带宽
5. **双通道分离设计**：ZMQ控制通道（轻量级） + Mooncake数据通道（RDMA）

## 双通道分离设计

```
控制通道（ZMQ）：
  ├─ 用途：metadata查询、完成通知
  ├─ 数据：几KB到几字节
  ├─ 优势：轻量级，不占用数据传输带宽
  └─ 时机：Layer 0查询 + 最后层通知

数据通道（Mooncake RDMA）：
  ├─ 用途：KV cache传输
  ├─ 数据：每层几十KB到几MB
  ├─ 优势：RDMA直接写入，低延迟高带宽
  └─ 时机：每层传输时（60层 × 8 ranks）
```

---

生成时间：2026-05-16
项目：vLLM-Ascend MooncakeLayerwiseConnector分析