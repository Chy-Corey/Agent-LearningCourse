## LangChain入门

通过之前的学习，我们了解到，一个Agent需要有以下组件：

1. 对应的LLM模型：接入Agent的大模型，作为Agent的大脑；
2. Tools：Agent可以使用的工具，作为Agent的手脚；
3. Prompts：发送给LLM的提示词，作为Agent的人设与指令，告诉大脑“你是谁”以及“你要按什么规则办事”；
4. Memory：Agent的记忆。

这里的Prompts值得展开，Agent具有哪些tool，保存了什么记忆，以及没有列出的Skills、MCP Server等等，都是通过Prompts发送给LLM的。Agent 框架的底层工作，本质上就是在每次对话前，高效、精准地组装这些组件，构成一段“超级提示词”。

在本章中，目标是展示使用LangChain开发一个幽默天气Agent的完整流程，因此我们只搭建以上四个组件，且结合代码学习。

```python
# 导入 dataclass 装饰器，用于快速定义带有默认值和类型提示的数据类（类似于定义数据结构）
from dataclasses import dataclass  
# 导入 LangChain 中用于创建智能体的核心工厂函数
from langchain.agents import create_agent
# 导入 tool 装饰器（把普通函数变成 AI 能用的工具）和 ToolRuntime（用于在工具中获取运行时上下文信息）
from langchain.tools import tool, ToolRuntime
# 导入内存型记忆存储器，用于保存对话的历史记录，让 AI 拥有“记忆”
from langgraph.checkpoint.memory import InMemorySaver
# 导入 DeepSeek 大语言模型的接口类
from langchain_deepseek import ChatDeepSeek
# 导入 dotenv 库，用于加载 .env 环境变量文件
from dotenv import load_dotenv  


# 加载环境变量中的DEEPSEEK_API_KEY
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

# 该工具为模拟获取用户地址
# runtime是LangChain自动注入到tool的运行时对象
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# 配置模型
model = ChatDeepSeek(
    model="deepseek-chat",	# 必须是LangChain支持的模型
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
# 实例化一个内存型的记忆存储器，作为短期记忆，用来在多次对话中保存聊天记录
# 大概流程为：每结束一次对话，存储对话信息；下一次在同一个对话ID发起对话时，携带保存的信息一起发送，实现多轮对话
checkpointer = InMemorySaver()

# 创建agent，组合以上搭建的模块（model，prompt，tool，memory）
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

response = agent.invoke(
    {"messages": [{"role": "user", "content": "今天天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(punny_response='佛罗里达今天依旧是阳光灿烂，真是"阳"光普照，心情也跟着"光"彩照人！
# 不过别忘了涂防晒，不然就要变成"佛罗里达烤人"了🌞😎', 
# weather_conditions='Florida总是阳光明媚！')

# 注意，我们可以使用相同的 `thread_id` 继续对话。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(punny_response='不客气！希望你的每一天都像佛罗里达的天气一样——"晴"天万里，"阳"光满溢！
# 下次再找我聊天气，我保证"气"象万千，妙语连"珠"！😄🌤️', 
# weather_conditions=None)


```



