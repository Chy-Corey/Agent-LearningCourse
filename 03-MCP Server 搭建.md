## MCP Server 搭建

> [Build an MCP server - Model Context Protocol](https://modelcontextprotocol.io/docs/develop/build-server)

我们已经知道，MCP server 的原语有：Prompt，Source和Tool。通过这三个功能，我们只需要搭建一次Server服务，就可以应用于所有支持 MCP 的客户端，本章将主要侧重于工具。

我们将参考[FlyAIBox/Agent_In_Action: Agentic AI 智能体开发实战](https://github.com/FlyAIBox/Agent_In_Action)项目中的第一个Demo，构建一个简单的 MCP 天气服务器，并将其连接到主机程序 DeepSeek。服务器将暴露两个Tool：

服务器实现提供了两个主要工具：

- **get_weather_warning**：获取指定城市ID或经纬度的天气灾害预警
- **get_daily_forecast**：获取指定城市ID或经纬度的天气预报

这些工具通过和风天气（QWeather）API 获取实时数据。然后我们将该服务器连接到 MCP 宿主（本例中为 DeepSeek）。

### 一、日志

在实现 MCP 服务器时，请务必小心处理日志：**对于基于 STDIO 的服务器：** 严禁写入标准输出 (stdout)。这包括：

- Python 中的 `print()` 语句
- JavaScript 中的 `console.log()`
- Go 中的 `fmt.Println()`
- 其他语言中类似的标准输出函数

写入 stdout 会破坏 JSON-RPC 消息并导致服务器崩溃。**对于基于 HTTP 的服务器：** 标准输出日志是可以的，因为它不会干扰 HTTP 响应。

写入日志时，参考以下两点：

1. 使用写入 stderr 或文件的日志库。
2. 对于 Python，要特别小心 - `print()` 默认写入 stdout。

```python
import sys
import logging

# ❌ Bad (STDIO)
print("Processing request")

# ✅ Good (STDIO)
print("Processing request", file=sys.stderr)

# ✅ Good (STDIO)
logging.info("Processing request")
```

### 二、环境准备

首先，让我们安装 `uv` 并设置我们的 Python 项目和环境：

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

完成后请务必重启终端，以确保 `uv` 命令生效。

现在，创建并设置项目：

```bash
# Create a new directory for our project
uv init weather
cd weather

# Create virtual environment and activate it
uv venv
.venv\Scripts\activate

# Install dependencies
uv add mcp[cli] httpx

# Create our server file
new-item server.py
```

### 三、搭建服务

首先初始化一个MCP服务器，然后实现对应的Tool函数即可，Tool函数用`@mcp.tool`包装。

#### 1. mcp.tool

`@mcp.tool()` 是 MCP（Model Context Protocol）Python SDK（特别是 `FastMCP` 库）中提供的一个装饰器。

它的核心作用非常简单：**将一个普通的 Python 函数，瞬间“变身”为 AI 模型可以识别和调用的标准 MCP 工具（Tool）**。可以把它理解为一个“魔法标签”，只要把它贴在函数上，这个函数就会被 MCP 服务端自动注册，并暴露给 AI 使用。

当你给一个函数加上 `@mcp.tool()` 后，它会在后台自动完成以下三件核心事情，把 Python 代码翻译成 AI 能看懂的“说明书”（即 JSON Schema）：

1. **提取工具名称（Name）**：直接把你的**函数名**作为 MCP 工具的标识符（比如函数叫 `get_weather`，工具名就是 `get_weather`）。
2. **生成工具描述（Description）**：自动读取你函数下方的**文档字符串（docstring，即 `"""..."""` 里的内容）**，作为工具的功能描述。AI 会根据这段描述来判断什么时候该调用这个工具。
3. **构建参数结构（Input Schema）**：通过分析你函数的**参数类型注解（Type Hints）**，自动生成一套严格的参数验证规则。AI 在调用时，必须按照这个规则提供参数（比如 `a: int` 告诉 AI 参数 `a` 必须是整数）。

**对比**：

没有 `@mcp.tool()` 时： 它只是一个普通的 Python 函数，只有你自己写的代码能调用它。

```python
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b
```

**加上 `@mcp.tool()` 后：** 它变成了一个 AI 工具。当 AI 需要算加法时，它就能在工具列表里发现这个函数，并按照要求传入参数来执行。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")

@mcp.tool()  # 👈 加上这个“魔法标签”
def add(a: int, b: int) -> int:
    """计算两个整数的和"""  # 👈 这段话会被 AI 看到，作为工具说明
    return a + b
```

#### 2. 代码

```python
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pathlib import Path
import os


# Initialize FastMCP server
mcp = FastMCP("weather")

# 常量，配置参数
# 加载 .env 文件中的环境变量
dotenv_path = Path(__file__).resolve().parents[0] / '.env'
load_dotenv(dotenv_path)

# 从环境变量中读取常量
QWEATHER_API_BASE = os.getenv("QWEATHER_API_BASE")
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")


# 工具函数（Tools）的辅助函数
def _normalize_base_url(raw_base: Optional[str]) -> str:
    """
    确保基础 URL 包含协议并以单个斜杠结尾，兼容 .env 中未写协议的情况
    """
    if not raw_base:
        raise RuntimeError("未配置 QWEATHER_API_BASE 环境变量")

    base = raw_base.strip()
    if not base.startswith(("http://", "https://")):
        base = f"https://{base.lstrip('/')}"

    # urljoin 要求目录风格以斜杠结尾，避免 'v7/weather/7d' 被覆盖
    if not base.endswith("/"):
        base = f"{base}/"

    return base


# 自检
try:
    _QWEATHER_BASE_URL = _normalize_base_url(QWEATHER_API_BASE)
except RuntimeError as err:
    print(f"[配置错误] {err}")
    _QWEATHER_BASE_URL = None


async def make_qweather_request(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    向和风天气 API 发送请求

    参数:
        endpoint: API 端点路径（不包含基础 URL）
        params: API 请求的参数

    返回:
        成功时返回 JSON 响应，失败时返回 None
    """
    if not _QWEATHER_BASE_URL:
        print("QWEATHER_API_BASE 未正确配置，已跳过请求。")
        return None

    if not QWEATHER_API_KEY:
        print("QWEATHER_API_KEY 未设置，已跳过请求。")
        return None

    safe_endpoint = endpoint.lstrip("/")
    url = urljoin(_QWEATHER_BASE_URL, safe_endpoint)

    # 使用 Header 方式认证（和风天气的新版本API）
    headers = {
        "X-QW-Api-Key": QWEATHER_API_KEY
    }

    async with httpx.AsyncClient() as client:
        try:
            print(f"请求 URL: {url}")
            print(f"请求参数: {params}")
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            print(f"响应状态码: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            print(f"响应内容: {result}")
            return result
        except httpx.HTTPStatusError as e:
            print(f"HTTP 状态错误: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"API 请求错误: {type(e).__name__}: {e}")
            return None


def format_warning(warning: Dict[str, Any]) -> str:
    """
    将天气预警数据格式化为可读字符串

    参数:
        warning: 天气预警数据对象

    返回:
        格式化后的预警信息
    """
    return f"""
预警ID: {warning.get('id', '未知')}
标题: {warning.get('title', '未知')}
发布时间: {warning.get('pubTime', '未知')}
开始时间: {warning.get('startTime', '未知')}
结束时间: {warning.get('endTime', '未知')}
预警类型: {warning.get('typeName', '未知')}
预警等级: {warning.get('severity', '未知')} ({warning.get('severityColor', '未知')})
发布单位: {warning.get('sender', '未知')}
状态: {warning.get('status', '未知')}
详细信息: {warning.get('text', '无详细信息')}
"""


def format_daily_forecast(daily: Dict[str, Any]) -> str:
    """
    将天气预报数据格式化为可读字符串

    参数:
        daily: 天气预报数据对象

    返回:
        格式化后的预报信息
    """
    return f"""
日期: {daily.get('fxDate', '未知')}
日出: {daily.get('sunrise', '未知')}  日落: {daily.get('sunset', '未知')}
最高温度: {daily.get('tempMax', '未知')}°C  最低温度: {daily.get('tempMin', '未知')}°C
白天天气: {daily.get('textDay', '未知')}  夜间天气: {daily.get('textNight', '未知')}
白天风向: {daily.get('windDirDay', '未知')} {daily.get('windScaleDay', '未知')}级 ({daily.get('windSpeedDay', '未知')}km/h)
夜间风向: {daily.get('windDirNight', '未知')} {daily.get('windScaleNight', '未知')}级 ({daily.get('windSpeedNight', '未知')}km/h)
相对湿度: {daily.get('humidity', '未知')}%
降水量: {daily.get('precip', '未知')}mm
紫外线指数: {daily.get('uvIndex', '未知')}
能见度: {daily.get('vis', '未知')}km
"""


# 借助辅助函数，实现工具函数
@mcp.tool()
async def get_daily_forecast(location: Union[str, int], days: int = 3) -> str:
    """
    获取指定位置的天气预报

    参数:
        location: 城市ID或经纬度坐标（经度,纬度）
                例如：'101010100'（北京）或 '116.41,39.92'
                也可以直接传入数字ID，如 101010100
        days: 预报天数，可选值为 3、7、10、15、30，默认为 3

    返回:
        格式化的天气预报字符串
    """
    # 确保 location 为字符串类型
    location = str(location)

    # 确保 days 参数有效
    valid_days = [3, 7, 10, 15, 30]
    if days not in valid_days:
        days = 3  # 默认使用3天预报

    params = {
        "location": location,
        "lang": "zh"
    }
    # 和风天气API文档 https://dev.qweather.com/docs/api/weather/weather-daily-forecast/
    endpoint = f"v7/weather/{days}d"
    data = await make_qweather_request(endpoint, params)

    if not data:
        return "无法获取天气预报或API请求失败。"

    if data.get("code") != "200":
        return f"API 返回错误: {data.get('code')}"

    daily_forecasts = data.get("daily", [])

    if not daily_forecasts:
        return f"无法获取 {location} 的天气预报数据。"

    formatted_forecasts = [format_daily_forecast(daily) for daily in daily_forecasts]
    return "\n---\n".join(formatted_forecasts)


@mcp.tool()
async def get_weather_warning(location: Union[str, int]) -> str:
    """
    获取指定位置的天气灾害预警

    参数:
        location: 城市ID或经纬度坐标（经度,纬度）
                例如：'101010100'（北京）或 '116.41,39.92'
                也可以直接传入数字ID，如 101010100

    返回:
        格式化的预警信息字符串
    """
    # 确保 location 为字符串类型
    location = str(location)

    params = {
        "location": location,
        "lang": "zh"
    }

    data = await make_qweather_request("v7/warning/now", params)

    if not data:
        return "无法获取预警信息或API请求失败。"

    if data.get("code") != "200":
        return f"API 返回错误: {data.get('code')}"

    warnings = data.get("warning", [])

    if not warnings:
        return f"当前位置 {location} 没有活动预警。"

    formatted_warnings = [format_warning(warning) for warning in warnings]
    return "\n---\n".join(formatted_warnings)


# 运行服务
def main():
    # Initialize and run the server
    print("正在启动 MCP 天气服务器...")
    print("提供工具: get_weather_warning, get_daily_forecast")
    print("使用 Ctrl+C 停止服务器")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

```

可以使用支持接入MCP Server的大模型应用测试，在这里我们先不测试，等搭建好客户端一起测试。

启动服务器：

```bash
python server.py
```

重新激活环境：

```bash
.venv\Scripts\activate
```

