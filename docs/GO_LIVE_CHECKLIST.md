# TFT Helper 上线清单

本清单用于生产环境部署前最后核对，覆盖：
- 服务端启动方式（含 `PORT`）
- 必需/可选环境变量
- API 列表与示例请求
- 域名、回调、权限配置点

## 1. 服务端启动（含 PORT）

说明：
- 项目本身不自动读取 `PORT` 环境变量。
- 生产启动时请在启动命令里显式把 `PORT` 传给 `uvicorn --port`。

### Linux/macOS

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export PORT=8000
python -m uvicorn app.main:app --app-dir src --host 0.0.0.0 --port "${PORT}"
```

### Windows PowerShell

```powershell
cd backend
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:PORT = "8000"
python -m uvicorn app.main:app --app-dir src --host 0.0.0.0 --port $env:PORT
```

### 启动后验证

- 健康检查：`GET /health` 返回 `{"ok": true}`
- Swagger：`/docs` 可访问
- 日志中看到缓存预热启动（`app.cache`）

## 2. 环境变量

## 2.1 必需环境变量（当前版本）

当前后端无 DB、无外部密钥、无必填模型路径环境变量。  
也就是说：`必需环境变量 = 无`。

## 2.2 可选环境变量（建议配置）

- `CACHE_PERSIST_ENABLED`
  - 默认：`1`
  - 作用：是否启用缓存快照落盘
- `CACHE_PERSIST_PATH`
  - 默认：`backend/.cache/cache_snapshot.json`
  - 作用：缓存快照文件路径
- `CACHE_REFRESH_INTERVAL_SECONDS`
  - 默认：空（按每日 00:00 刷新）
  - 作用：测试时可改为固定秒数刷新，如 `60`
- `TESSDATA_PREFIX`
  - 默认：自动探测
  - 作用：`pytesseract` 的语言模型目录（若你走 tesseract 方案）
- `ASSET_SYNC_WORKERS`
  - 默认：脚本内默认值
  - 作用：`tools/sync_augment_assets.py` 并发下载数

## 3. API 列表与示例请求

基地址示例：`https://api.example.com`

### 3.1 核心查询接口

- `GET /health`
- `GET /api/champions`
- `GET /api/champions/{championId}/augments`
- `GET /api/champions/{championId}/augment-combos`
- `GET /api/augments/mapping`

示例：

```bash
curl -X GET "https://api.example.com/health"
curl -X GET "https://api.example.com/api/champions?sort=winRate&order=desc&limit=20&offset=0"
curl -X GET "https://api.example.com/api/champions/887/augments?sort=winRate&order=desc"
curl -X GET "https://api.example.com/api/champions/887/augment-combos"
curl -X GET "https://api.example.com/api/augments/mapping"
```

### 3.2 实战分析/OCR接口

- `POST /api/battle/recommend`
- `POST /api/ocr/augments-names`（`multipart/form-data` 上传图片）
- `POST /api/augments/resolve-names`

示例：

```bash
curl -X POST "https://api.example.com/api/battle/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "championId": 887,
    "round": 2,
    "pickedAugmentIds": [1234],
    "candidateAugmentIds": [5678, 9101, 1112]
  }'
```

```bash
curl -X POST "https://api.example.com/api/augments/resolve-names" \
  -H "Content-Type: application/json" \
  -d '{"names":["尤里卡","珠光护手","炼狱导管"]}'
```

```bash
curl -X POST "https://api.example.com/api/ocr/augments-names" \
  -F "image=@/path/to/screenshot.jpg"
```

完整接口说明见：`docs/API.md`

## 4. 域名 / 回调 / 权限配置

## 4.1 小程序域名配置（必须）

微信小程序后台需配置以下合法域名（生产环境必须 HTTPS）：
- `request` 合法域名：`https://api.example.com`
- `uploadFile` 合法域名：`https://api.example.com`（OCR 上传用）
- `downloadFile` 合法域名：`https://api.example.com`（如使用下载能力）

说明：
- 开发工具里“**不校验合法域名**”只用于本地调试，不能替代生产配置。

## 4.2 服务端出网权限（必须）

后端需要能访问：
- `https://hextech.dtodo.cn`（英雄数据）
- `https://raw.communitydragon.org`（海克斯映射）

若出网被防火墙限制，会导致数据更新失败或 OCR 字典不完整。

## 4.3 回调配置

当前版本无第三方回调 URL（Webhook/OAuth 回调）要求。

## 4.4 端口与网络策略

- 对外开放：`PORT`（默认可用 `8000`）
- 反向代理建议开启：
  - HTTPS
  - Gzip/Brotli
  - 合理超时（OCR 上传建议读超时 >= 60s）

## 4.5 静态资源缓存

`/static/*` 已返回：
- `Cache-Control: public, max-age=31536000`

上线核对：
- 响应头是否生效
- CDN 是否尊重源站缓存头

## 5. 上线前验收（建议逐项勾选）

- [ ] 后端可用 `PORT` 启动并对外监听 `0.0.0.0`
- [ ] `/health` 200，`/docs` 可访问
- [ ] `/api/champions`、`/api/champions/{id}/augments`、`/api/champions/{id}/augment-combos` 返回正常
- [ ] `/api/battle/recommend` 返回推荐结果
- [ ] `/api/ocr/augments-names` 上传图片可返回 `left/middle/right`
- [ ] 小程序生产域名配置完成（request/uploadFile/downloadFile）
- [ ] 后端具备访问 `dtodo` 与 `communitydragon` 的出网权限
- [ ] 静态资源缓存头生效（`Cache-Control`）
