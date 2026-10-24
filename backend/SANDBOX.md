# CodeWizard 代码沙箱

CodeWizard 不再在 FastAPI 进程中执行模型生成的 Python。项目启用代码沙箱后
固定使用 Docker + gVisor；未配置沙箱或 gVisor runtime 时，代码执行会被拒绝，
不会降级到 E2B、进程内 `exec` 或普通 Docker `runc`。

## 启用 Docker + gVisor

gVisor 的 `runsc` 需要安装在 Linux Docker 主机上。先构建只包含数据分析依赖的镜像：

```bash
docker build -t industry-code-sandbox:latest ./sandbox
docker run --rm --runtime=runsc industry-code-sandbox:latest
export CODE_SANDBOX_PROVIDER=docker-gvisor
```

此后每次执行都会创建一个一次性容器，默认断网、只读根文件系统、非 root 用户，
并限制 CPU、内存、进程数、超时和输出大小。`runsc` 不存在时执行会失败，系统不会
自动降级到普通 Docker 运行时。

`CODE_SANDBOX_DOCKER_RUNTIME` 不能为空；空值会关闭代码执行，避免降级到 Docker
默认的 `runc`。

## 配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CODE_SANDBOX_PROVIDER` | `disabled` | `disabled` 或 `docker-gvisor`；项目示例配置已启用后者 |
| `CODE_SANDBOX_TIMEOUT_SECONDS` | `30` | 单次执行时间上限 |
| `CODE_SANDBOX_MAX_OUTPUT_CHARS` | `20000` | stdout/stderr 最大字符数 |
| `CODE_SANDBOX_MAX_CHART_BYTES` | `5242880` | 图表最大字节数 |
| `CODE_SANDBOX_MEMORY` | `512m` | Docker 内存上限 |
| `CODE_SANDBOX_CPUS` | `1.0` | Docker CPU 上限 |
| `CODE_SANDBOX_PIDS` | `64` | Docker 进程数上限 |
| `CODE_SANDBOX_DOCKER_IMAGE` | `industry-code-sandbox:latest` | 私有沙箱镜像 |
| `CODE_SANDBOX_DOCKER_RUNTIME` | `runsc` | Docker 安全运行时 |

不要给沙箱挂载项目目录、`.env`、Docker socket 或数据库目录，也不要把主服务的
API Key 传入沙箱。
