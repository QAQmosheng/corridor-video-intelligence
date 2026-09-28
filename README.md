# 管廊视频智能体独立原型

这是依据需求文档 10.4 重新设计的独立视频识别系统。运行代码不导入、不读取、不调用 `platformv2` 的模型、权重、配置或进程；WeKnora 也不是它的运行依赖。

## 一键体验

双击 `一键启动视频识别.bat`，浏览器会打开 `http://127.0.0.1:8765`。点击“开始演示”可动态查看：

1. 三台摄像头分别建立基准；
2. CAM-A 发现佩戴安全帽人员，冻结整个廊段的会话前基准；
3. CAM-B 同时登记工具箱，但人员尚未离场，所以不报警；
4. 人员跨镜移动到 CAM-B，仍属于同一廊段会话；
5. 全廊段无人并超过离场确认时间后，验证工具箱持续存在；
6. 只在 CAM-B 生成一条 `abandoned_object` 事件；
7. CAM-C 的火焰候选进入视觉复核队列，演示默认不调用外部大模型。

演示中的画面由代码实时生成，再由真实的 `NativeVisionPipeline` 分析，不是前端写死的检测结果。

## 命令行

```powershell
cd video-recognition-module
\.venv\Scripts\python.exe -m corridor_video.cli validate-config
\.venv\Scripts\python.exe -m corridor_video.cli simulate
\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

分析一个自有视频文件：

```powershell
\.venv\Scripts\python.exe -m corridor_video.cli analyze-video --source D:\videos\sample.mp4 --camera-id CAM-A --max-frames 1000
```

监控单路 RTSP（首次成功读取的画面采样为该连接的基准图；断流重连后原基准失效并重新采样）：

```powershell
\.venv\Scripts\python.exe -m corridor_video.cli monitor-rtsp `
  --source "rtsp://user:password@192.168.1.10:554/Streaming/Channels/101" `
  --camera-id CAM-01 --corridor-id CORRIDOR-01
```

该入口默认持续运行并输出 JSON Lines 事件；`Ctrl+C` 停止。现场联调可追加
`--max-frames 1000 --max-reconnects 5`。日志与最终统计会移除 RTSP 用户名、密码和查询参数。

启动带实时标注画面、当前报警和事件记录的持续检测页面：

```powershell
\.venv\Scripts\python.exe -m corridor_video.cli live `
  --source "D:\videos\sample.mp4" --camera-id CAM-01 --port 8766
```

浏览器打开 `http://127.0.0.1:8766`。本地视频默认按原始帧率循环播放；增加 `--no-loop` 可在文件结束后停止。
`--source` 也可以直接使用 RTSP 地址，断流时页面会显示重连状态。候选框绘制为黄色或青色，已激活报警使用红色粗框，
并在报警持续期间保持显示。

仓库内的 `启动19-2持续检测.bat` 已配置好 `platformv2/19-2video.mp4`，双击即可启动服务并打开页面。

这里的差异不是相邻帧帧差：第 1 张有效帧成为基准，后续每张当前帧都与该基准比较；只有差异面积连续达到
`stable_confirm_frames` 才进入目标检测。会话期间冻结基准，断流、镜头移动或全局画面突变时禁止覆盖。

报警与逐帧检测相互分离：相同类别且位置重叠的候选连续确认后只生成一次事件，持续期间不重复报警；连续
`lifecycle.clear_frames` 帧且达到 `clear_seconds` 后事件关闭，再次出现才允许生成新事件。相关参数位于 `config/default.toml` 的
`[lifecycle]` 段。

若没有现成虚拟环境，可使用 Python 3.11+ 创建环境并执行 `pip install -e .`。原型只依赖 NumPy 与 OpenCV，不依赖 GPU。

## 代码结构

- `src/corridor_video/vision.py`：新的视频算法入口，完成动态基准、差异门控、可解释候选分类和质心跟踪；
- `src/corridor_video/engine.py`：廊段级会话、跨摄像头离场判定、遗留物三条件和事件去重；
- `src/corridor_video/vlm.py`：可选的火烟/积水视觉复核边界；
- `src/corridor_video/outbox.py`：SQLite 事件 Outbox，保证 WeKnora 不可用时事件不丢失；
- `src/corridor_video/demo.py` 与 `web/`：动态原型服务和前端；
- `config/default.toml`：独立摄像头拓扑与全部阈值；
- `schemas/video_event.schema.json`：内部事件契约；`schemas/weknora_event.schema.json`：10.6 对外契约；
- `docs/ALGORITHM_DESIGN.md`：原理、边界与生产化路线；
- `docs/WEKNORA_INTEGRATION.md`：接入说明。

## 原型边界

当前语义层采用可解释的颜色、几何与位置特征，作用是把 10.4 的完整计算链路和状态机先跑通，不应直接作为生产准确率结论。生产阶段应使用本项目自己的标注集训练新检测器，导出 ONNX 后替换 `semantic_candidates()`；动态基准、跟踪、廊段会话、事件规则、Outbox 和接口无需重写。
