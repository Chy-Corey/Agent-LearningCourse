from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv


# 加载环境变量中的api_key
load_dotenv()

# 定义系统提示词
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""


# 定义上下文模式
# 使用 dataclass 定义一个自定义的上下文数据结构，用于在程序运行时给智能体传递额外的状态信息
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str


# 定义工具
# 该工具为模拟提供天气api
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# 配置模型
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.5,
    max_tokens=2000,
    timeout=None,
    max_retries=3,
    # api_key="...",
    # other params...
)


# 定义响应格式
# 定义 AI 最终回复时必须遵守的数据结构（强制 AI 输出 JSON 格式的数据，而不是随意的文本）
@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None


# 设置记忆
# 实例化一个内存型的记忆存储器，用来在多次对话中保存聊天记录
checkpointer = InMemorySaver()

# 创建agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# 运行代理
# `thread_id` 是给定对话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

# 使用 stream 方法，并设置 stream_mode="values" 来获取每一步的状态快照
for step in agent.stream(
        {"messages": [{"role": "user", "content": "今天天气怎么样？"}]},
        config=config,
        context=Context(user_id="1"),
        stream_mode="values"
):
    # 获取当前步骤中的最后一条消息
    last_message = step["messages"][-1]

    # 打印出消息内容（如果是工具调用，会显示 tool_calls；如果是工具执行结果，会显示返回内容）
    print(f"--- 当前步骤 ---\n{last_message}\n")

# 注意，我们可以使用相同的 `thread_id` 继续对话。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="你真是'雷'厉风行地欢迎！帮助你保持'当前'天气总是'轻而易举'。我只是'云'游四方，等待随时'淋浴'你更多预报。祝你在佛罗里达的阳光下度过'sun-sational'的一天！",
#     weather_conditions=None
# )
