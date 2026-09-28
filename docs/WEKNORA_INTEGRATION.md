# 接入 WeKnora 说明

## 推荐架构

视频模块保持独立进程，负责取流、推理、跨镜会话、证据保存和可靠投递；WeKnora 负责事件接收、知识检索、处置建议、人工确认和工单编排。WeKnora 不加载视频模型，视频推理线程也不等待 WeKnora。

```text
摄像头 → video-recognition-module → SQLite Outbox → HTTP 事件适配器 → WeKnora
                    ↓                                      ↓
                本地证据库                         知识检索与处置智能体
```

## WeKnora 需要提供的入口

```http
POST /api/v1/events
Content-Type: application/json
Idempotency-Key: <event_id>
Authorization: Bearer <token>
```

请求体采用 `schemas/weknora_event.schema.json`；模块内部事件通过 `delivery.to_weknora_payload()` 转换为第 10.6 节要求的 `event_time`、`location`、`algorithm_confidence` 和 `upstream_metadata` 结构。WeKnora 必须先做 Schema 校验，再以 `event_id` 建唯一索引并返回：

```json
{
  "accepted": true,
  "event_id": "VID-...",
  "internal_event_id": "EVT-...",
  "duplicate": false
}
```

重复请求返回原 `internal_event_id`，不得重复创建处置任务。视频模块收到成功响应后，把 Outbox 记录标为 `sent`；网络失败或 5xx 使用指数退避，4xx Schema 错误进入死信队列并告警。

## 当前仓库的具体改造位置

已核对 `GL2Q/WeKnora` 当前 Go 工程。建议沿用现有分层，不把视频状态机塞入聊天 Handler：

| 层 | 建议文件/位置 | 职责 |
| --- | --- | --- |
| 数据模型 | `internal/types/video_event.go` | 事件、图片、检测框和接收结果类型 |
| 接口 | `internal/types/interfaces/video_event_service.go` | 定义接收、查询、状态推进能力 |
| 仓储 | `internal/application/repository/video_event.go` | `event_id` 幂等写入与状态读取 |
| 服务 | `internal/application/service/video_event.go` | Schema/白名单校验、事件映射、触发处置流程 |
| HTTP | `internal/handler/video_event.go` | 读取 `Idempotency-Key`，返回 201/200/409/422 |
| 路由 | `internal/router/routes_video_event.go` | 注册 `POST /events`，由 `router.go` 的 `/api/v1` 组调用 |
| 依赖注入 | `internal/container/container.go` | 注册 repository、service、handler 构造器，并加入 `RouterParams` |
| 数据库 | `migrations/versioned/` 与 `migrations/sqlite/` | 同步增加 PostgreSQL/SQLite 两套版本迁移 |

该路由应位于现有认证中间件之后。视频服务使用租户级 API Key，并在 API-key 权限矩阵中增加最小的 `ingest_video_events` 能力，不复用管理员全权限 Key。当前 CORS `AllowHeaders` 未包含 `Idempotency-Key`；若未来由浏览器直传需要补充，独立视频进程的服务端 HTTP 调用则不受 CORS 影响。

## 事件映射

| 视频事件 | WeKnora 语义 | 默认处置 |
| --- | --- | --- |
| `person_intrusion` | 未正确佩戴安全帽人员 | 进入人员异常规则 |
| `worker_detected` | 外观上为正确佩戴安全帽人员 | 状态记录，不能视为授权 |
| `fire_smoke` | 视觉复核确认火焰/烟雾 | 进入消防事件流程 |
| `waterlogging` | 视觉复核确认积水 | 进入积水流程，水深保持 unknown |
| `abandoned_object` | 全廊段离场后持续存在的新增物 | 进入遗留物流程 |
| `other_intrusion` | 动物或其他非人员目标 | 进入通用入侵流程 |
| `visual_review_required` | 候选存在但复核不可用 | 人工复核，不自动派高风险工单 |

## 证据处理

原型事件可保存本地文件元数据。正式接入时建议先上传到统一对象存储，再在事件中放短期签名 URL、SHA-256、媒体类型、尺寸和采集时间。WeKnora 不应接受任意本地路径，也不应长期保存可公开访问的 URL。

## 分阶段接入

1. 在 WeKnora 新增事件表、唯一索引、Schema 校验和接收路由；
2. 配置服务账户和最小权限令牌，密钥仅从环境变量读取；
3. 增加独立的 Outbox 投递工作线程，启用超时、指数退避、熔断和死信；
4. 以 `enabled=false` 保持正式投递关闭，先做本地回放与契约测试；
5. 开启影子投递，在 WeKnora 中只记录不派单，对照人工标注评估；
6. 小范围摄像头灰度启用，验证幂等、证据访问、告警恢复和回滚；
7. 达到验收门槛后扩大范围，并保留按摄像头一键回退到“仅记录”模式。

## 接入准备状态

当前模块已经具备稳定事件对象、JSON Schema、SQLite Outbox、可配置 WeKnora 端点和 HTTP 客户端边界。正式接入前仍需由 WeKnora 侧实现 `/api/v1/events`，并确定身份认证、对象存储与事件表结构；这些属于接入阶段，不应耦合进视频算法内核。
