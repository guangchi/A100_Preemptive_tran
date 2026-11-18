# P 端入口固定窗口（2026-11-17）

本次修改实现用户确认的语义：**P HTTP 入口收到首条请求 → 开启固定 5ms 窗口 → 收集这 5ms 内实际到达的请求 → 到期后排序、整窗提交 → 引擎保留窗口顺序，再按计算预算执行。** 本次只修改实现并做 CPU 功能验证，没有运行 GPU 推理或性能实验。

## 窗口与排序

- 计时位置：P 服务的 ASGI HTTP 入口，读取请求体、JSON 解析、tokenization 和 EngineCore 输入排队之前。使用 `time.monotonic_ns()`。
- 每个窗口区间为 `[首条到达时间, 首条到达时间 + 5ms)`；到达时间恰好等于 deadline 的请求开启下一窗。后续请求不延长当前窗，事件循环定时器迟到也不改变归属。
- 单条请求也等待满 5ms，删除旧的 idle singleton 提前放行逻辑。没有请求时不开空窗。
- 窗内按引擎预处理前已经确定的实际 `prompt_token_ids` 数量升序排序，等长按入口到达顺序稳定排列。窗间 FIFO，不跨窗重排或合并。
- 请求体或 tokenization 尚未就绪时，保留它原有的窗口归属。该窗等待这些成员就绪，后窗不能越过它；因此 5ms 是收集边界，不是“所有情况下必须在第 5ms 完成提交”的保证。无效、失败或取消的 HTTP 请求会移除等待资格，避免阻塞其余请求。

例如三次突发各 5 条，每次的 5 条都在同一入口 5ms 区间内，批间隔 25/50ms，即使引擎忙，也保留 5+5+5 三个窗口。**窗口按时间划分，不硬编码每窗必须 5 条**：实际入口到达跨过 deadline 时仍会分窗。

## 整窗提交与引擎执行

`pd_ingress_window.py` 在 API 事件循环中收集并排序整窗，使用 vLLM 的 `MsgpackEncoder` 编码原始 `EngineCoreRequest`，保留 sampling、请求 ID、NIXL KV transfer 参数等字段。整个窗口作为一条现有 UTILITY 消息提交给 EngineCoreProc 的 `pd_add_ingress_window` 方法。

该方法在 **EngineCore 主循环**内解码、预处理并逐条注册整窗。方法返回之前不会进入下一次 `schedule()`，所以引擎不会只收到第一条就提前开始计算。注册失败时回滚已尝试注册的成员；API 侧同时清理已注册的输出 collector。取消发生在提交中时，还会在整窗确认后补发 abort，防止 abort 先于 ADD 导致请求残留。

`pd_local_sort_scheduler.py` 保留原有启动类名，改为普通 FCFS 顺序消费，不再包含调度器入窗、后台发布线程或 singleton 快捷路径。已经运行的计算和 vLLM 的 token/KV 预算、chunked prefill 继续工作。**整窗原子接纳不等于整窗必须在一次 GPU 执行步中算完**；例如长请求仍可按当前 2048-token 预算分步计算。

## 接入与开关

`run_pd_pair.sh` 在 `PD_LOCAL_SORT=1` 时为 P 设置 `PD_INGRESS_WINDOW=1`，并载入 `diagnostic_bootstrap/sitecustomize.py` 的生产导入钩子。该路径独立于 `PD_PIPELINE_TRACE`，不需要打开诊断采集。D 设置 `PD_INGRESS_WINDOW=0`。没有修改共享安装的 vLLM。

现有两个开关继续独立：

| 启动方案 | P 入口窗口排序 | 传输抢占 |
|---|---:|---:|
| `baseline` | 关 | 关，原生整请求传输 |
| `stream_fifo` | 关 | 关，流式 FIFO |
| `sort_only` | 开 | 关 |
| `preempt_only` | 关 | 开 |
| `new` | 开 | 开 |

仍可用 `PD_LOCAL_SORT=0/1`、`PD_PRIORITY_PREEMPTION=0/1` 独立覆盖。启动脚本固定窗口为 5ms。

当前入口窗口路径适配本项目的单 API 进程、DP=1、`POST /v1/completions`、单文本或单 token-ID prompt、`n=1`。请求级多 prompt、best_of/n>1 和 beam search 在入口返回明确的 400，避免顺序生成多个引擎请求时等待整窗造成死锁。多模态、prompt embeddings、resumable、structured output 和其他绕过该 HTTP 入口的生成调用不支持此窗口路径，会明确报错。关闭排序时使用原始生成接纳路径。

## 可审计信息与验证

- 常规 P 日志：`PD ingress admission: window=... rank=... req=... tokens=... admitted_ns=... deadline_ns=... released_ns=...`。
- 可选 trace：新增 `ingress_http`（含共享 HTTP ID 和入口时间）、`ingress_window_submit`、`ingress_window_registered`，保留每个请求的 `window_admit/window_release`，增加 `source=p_http_ingress`。新路径以整窗 utility 入队，不会伪造旧的逐条 ADD 队列事件。
- `test_pd_admission_deadline.py` 覆盖固定窗口、满 5ms 单请求、deadline 边界、忙引擎下 5+5+5 与稳定排序、迟到请求体、无效请求、提交前后取消、失败回滚、关闭清理；使用真实 vLLM 请求序列化、UTILITY 分发、AsyncMPClient 与输出 collector 清理做 CPU 集成验证。
- 另检查真实 vLLM API app 构造、trace 与生产钩子共存、排序关闭时的原生导入路径，并运行已有 router/流式协议/trace/恢复测试及脚本语法检查。没有启动 P/D 推理服务。

最终上述 6 个测试模块共 42 项 CPU 检查通过，Python 编译和 `bash -n run_pd_pair.sh` 通过。4 种导入配置（P 开窗无 trace、P 开窗带 trace、关闭排序无 trace、D 仅 trace）检查通过；带 trace 的真实 vLLM API app 构造通过。

以前 C8/C16/C32 与入窗诊断的性能数据均属于旧的“调度器入窗 + singleton 提前放行”实现，保留作历史结果，不能代表本次入口窗口实现的性能。新的 TTFT、吞吐或抢占收益需要之后单独重测。

后续已于2026-11-17在GPU4/5完成三轮四组GPU重测，见[完整报告](../results/nvlink-ingress-window-gpu4-5-20261117-015530/REPORT.md)与技术报告第23节。89个正式入口窗口通过固定边界、排序与整窗接纳检查；实际同批到达跨度超过5ms，因此常见小窗而非固定5条一窗。具体TTFT及25/50ms差异以该报告为准。


2026-11-17 入口时序补充：已完成GPU4/5先50ms后25ms的三次重启、六组只排序正式对照。入口JSON/逐token校验CPU成本与ASGI任务调度会改变5ms窗成员，不支持单纯首次运行解释；同样代码CPU耗时波动的底层原因尚未唯一定位。新增可选`PD_API_TIMING=1`细分观测，不修改窗口算法。详见[顺序复测报告](../results/nvlink-ingress-order-50first-gpu4-5-20261117-030213/REPORT.md)。

## 收集阶段与预处理阶段分离（2026-11-17，代码优化）

此前入口在登记第一条请求后，立即读取请求体、解析JSON并逐token校验，再进入FastAPI；这些同步工作与其他HTTP任务共用事件循环线程，可能阻塞后续成员登记。当前改为：

`轻量登记到达/窗口 → 异步等待本窗固定5ms截止 → 窗内各请求开始API预处理 → 全部有效成员准备完毕 → 稳定排序 → 单条UTILITY整窗提交`

收集阶段不读取请求体、不启动本窗成员的JSON或模型校验。等待截止时让出事件循环，允许其他已经就绪的HTTP任务入窗；定时器提前触发仍重新检查deadline，到达恰好等于deadline仍进入下一窗。截止后各成员在各自HTTP协程中预处理，等待全部有效成员就绪，不逐条提交EngineCore。原有取消、失败清理与窗间FIFO保留，关闭时也释放尚在等待收集截止的HTTP协程。

删除入口的body缓冲/重放、重复`json.loads`和逐token `all(isinstance(...))`。请求体直接交给Starlette/FastAPI读取并解析一次；多prompt、n/best_of和beam限制移至`OpenAIServingCompletion.render_completion_request`之前，复用已校验的`CompletionRequest`。该模型已验证prompt为同质的token列表、文本或多prompt列表，附加的单prompt检查只检查列表首元素与固定字段，不再次遍历全部token。无效JSON及token类型/范围错误仍由FastAPI返回原有校验错误；多prompt等仍在任何引擎生成前拒绝，防止顺序fanout等待整窗时死锁。生产校验钩子不依赖trace开关。

这里的批量是“先收集、再准备整窗、最后一次提交”。JSON/Pydantic仍逐个HTTP请求执行；没有加入多进程或线程池，也没有宣称多个CPU密集任务会因`asyncio`自动并行。EngineCore中涉及Request/KV状态的注册仍由主循环顺序执行，整个窗口注册完前不调度计算。

这个改动避免**同一窗口成员自己的预处理**占用收集时间，并不完全隔离前一窗口的API处理、HTTP协议处理或上游router；计时位置仍是ASGI应用入口而非内核收包。因而不能保证客户端同批5条永远同窗，也没有把固定时间窗改成固定5条批次。若要求接收线程完全独立于预处理，还需独立进程等更进一步的架构改动。

细分trace继续保留body读取与`api_ingress_forward`观测，新增`ingress_preprocess_start`；旧`ingress_json/ingress_validate`同步span随重复工作删除而消失，旧顺序复测分析脚本和历史数据应使用相应归档源码。新增收集等待可能增加提交前等待，性能净收益不能由CPU功能检查推断。本次没有启动GPU实验，之前的TTFT表仍属于优化前版本。

本次6个CPU测试模块共48项通过（入口/trace 26项，router/流式协议/metadata/恢复22项），并通过Python编译、带/不带trace的生产校验导入检查。新增验证包括：五条先登记再进行超出5ms的模拟预处理仍整窗提交、真实FastAPI+CompletionRequest单次JSON解析、无需再次遍历10000个token、多prompt/n/beam拒绝、原始422错误不阻塞正常请求，以及收集期间取消和提前关闭清理。未启动模型服务或创建GPU推理上下文。

## 以排序为目标的流程复查（2026-11-17，仅CPU验证）

重新确认排序发生于API侧、实际prompt token数全部就绪之后，等长保留入窗顺序；EngineCore使用FCFS保留该顺序。整窗UTILITY用于避免已排顺序在实际接纳时被部分提前执行破坏，不是要求GPU一次算完整窗，也没有第二个5ms调度窗口。窗内合法成员尚未准备好时仍需等待，才能保证未知长度的short不会被已准备好的long越过。

本次将router每请求创建的后端ClientSession改为应用生命周期连接池，复用keep-alive连接、统一退出关闭，并保持P/D并发与SSE行为；EngineCore对已排顺序的第二次sorted校验改为相邻键线性检查。入口JSON去重等上一轮优化保留，模型合法性检查不跳过。新增回归后6个CPU模块共51项通过。

实际进行了独立进程、真实HTTP的CPU功能检查，使用生产router、生产入口gate及真实FastAPI/CompletionRequest；P/D模型和引擎执行为CPU桩，CUDA设备不可见，没有模型推理或KV传输。25ms三批窗口构成为`1+4 / 5 / 5`，50ms为`5 / 5 / 5`。首批拆窗时router到首窗开始后5.318ms才发送第二条的P请求头，晚于5ms截止，首条预处理则5.708ms才开始，直接证明其拆窗发生在本窗预处理之前。其余五批P入口跨度3.354–3.684ms，均成功五条同窗。

因此当前代码**能够形成五请求窗口，但不保证所有客户端五请求批次同窗**。严格条件是五条实际P入口时间位于同一固定5ms区间；router、HTTP连接、前窗处理及调度仍会影响到达。若要求客户端每组五条无论到达时间都一起排序，需要另行定义显式批次语义，不能据此扩大或回填固定窗口。完整审查、原始CPU时间戳和验证边界见[流程复查报告](../results/ingress-flow-review-20261117-040748/REPORT.md)。

## 当前部署默认：10ms及GPU4/5重测（2026-11-17）

按用户要求，`run_pd_pair.sh`与生产app钩子的窗口默认值改为10ms，并允许用`PD_LOCAL_SORT_WINDOW_MS`覆盖。底层Gate/ASGI类仍保留5ms的显式CPU测试默认，历史文档中的5ms语义与数据不重写；实际服务启动以生产参数为准。本轮所有正式窗口均验证为10,000,000ns。

采用本页最近的去重、先收集后预处理及router连接池版本，物理GPU4(P)/5(D)、NVLink、只排序、C16/单槽/0.1ms poll。三次重启各预热50→25ms，正式50→25→50→25ms，共180条正式、90条预热全部通过输出校验。6次回放均值±回放样本标准差(ms)：

|间隔|Short|Medium|Long|全部|
|---|---:|---:|---:|---:|
|25ms|2344.41±148.30|2477.74±239.63|3141.75±9.17|2583.70±121.31|
|50ms|2315.58±52.91|2414.21±29.41|3060.73±22.96|2534.01±32.54|

25/50ms分别10/18、6/18个客户端批次完整同窗；整次严格5+5+5分别1/6、0/6，10ms仍不能保证每批五条同窗。最后一组25ms为5+5+5，Short2061.77ms，比配对50ms快353.28ms；05/06同窗让06先调度，是其中一条直接原因。两场景Short平均TTFT差−28.83ms，但以首条客户端发起为共同原点，首文本均值仅差−3.48ms，故不能把均值差全归为处理提速。

55个正式窗口边界/排序/接纳、270条流式请求3370对WRITE/DONE与18次NVLink payload计数全部通过；实验上下文仅4/5且已释放。与旧5ms GPU数据相比还包含多项代码优化，不能视为只改变窗长的严格消融，本轮未跑baseline/只抢占/组合。逐轮表、原始成员和软件阶段见[10ms完整报告](../results/nvlink-ingress-window10-gpu4-5-20261117-080249/REPORT.md)。

## Router解码与发送编码优化（2026-11-17，仅CPU验证）

在10ms GPU实验之后，进一步优化router：保留异步读取，使用已安装的orjson从bytes解码；token数组只编码一次、复用到P/D不同元数据，提前准备两份载荷，aiohttp以data=bytes发送，避免P/D任务各自标准JSON编码。连接池复用、共享请求ID、KV参数、P/D并发和SSE行为保留。没有新增按客户端批次凑齐五条的屏障，固定窗口仍以实际P入口到达为准。

新增router_read_body_span、router_decode_span、router_prepare_span时间戳，区分I/O读取、解码与构造编码。CPU同步代码仍在单事件循环中执行，没有声称线程/多核隔离。当前运行环境需orjson（已存在，无安装改动），非法JSON及非对象/list形状输入400，超限body保留413。

6个CPU模块共54项通过。一次独立进程真实HTTP CPU桩检查共30条，25/50ms各三批均形成5+5+5；P入口跨度分别5.495/2.328/2.563ms与2.050/1.187/4.768ms。router解码wall均值0.086ms、最大0.619ms，构造P/D字节载荷均值0.055ms、最大0.347ms。本次不是优化前后性能配对，也没有GPU模型负载，不能据此保证真实GPU下全部同窗或TTFT改善。详见[router优化报告](../results/router-preprocess-review-20261117-081626/REPORT.md)。


## 最新router四方案GPU4/5重测（2026-11-17，已完成）

已用当前router版本、10ms固定窗、C16/单槽/0.1ms poll完成三轮baseline、只排序、只抢占、组合×25/50ms对照。24次正式360条和24次预热360条均通过输出校验；只排序/组合12次正式均为5+5+5，36个窗口边界/排序/原子接纳通过，没有拆窗或跨批合窗，6组排序/组合窗口成员和顺序完全相同。本轮最大同批入口跨度9.482ms，仍不能保证其他负载永远同窗，也不是旧router与新router的严格配对。

组合Short TTFT为25ms 2055.37±7.34ms、50ms 2011.97±6.88ms，相对baseline降低21.05%/21.70%；只排序为2103.70±27.62/2047.19±12.44ms，只抢占为2750.12±24.20/2713.63±11.08ms，baseline为2603.51±17.18/2569.49±31.48ms。排序主要减少Short约679–683ms的首次P调度前等待；组合把可发Short等待WRITE由约57–59ms降至约5ms。只抢占未重排计算，流式发布/event/writer额外阶段仍抵消插队收益；组合Long比baseline慢约9.3%，平均TTFT与makespan需分别看待。

两间隔窗口成员相同，50ms Short较低包含约25ms的发起时刻后移，其余变化见逐阶段分析；不能再用本轮拆窗或唯一的拥堵原因解释。48次NVLink payload计数、540条流式请求6732对WRITE/DONE、最终通知全部通过；运行源码未变，实验上下文仅4/5，两卡已释放。完整分类TTFT、逐轮/E2E/窗口/仲裁数据及分析见[最新router四方案报告](../results/nvlink-router-window10-fourway-gpu4-5-20261117-082408/REPORT.md)和技术报告第29节；本节数据取代CPU桩结果作为本轮真实GPU观察。
