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

LangChain 能够把工具（Tools）的信息巧妙地融入提示词（Prompt），并精准判断大语言模型（LLM）是否想调用工具，主要依赖于一套**“工具描述注入 + 结构化输出（Function Calling）”**的机制。

具体可以分为以下两个核心步骤来理解：

### **1. LangChain 是如何将 Tools 的信息加到 Prompt 里的？**

当你使用 `create_agent` 或 `bind_tools` 时，LangChain 会在后台自动执行一系列操作，把工具变成 LLM 能看懂的“说明书”：

- 提取工具元数据：

   LangChain 会通过你定义工具的 Python 函数，自动提取三个关键信息：

  - **工具名称：** 默认取自函数名（也可以通过 `@tool("custom_name")` 指定）。
  - **工具作用：** 默认取自函数的文档字符串（即 `"""..."""` 里的内容）。
  - **输入参数：** 通过 Python 的类型提示（如 `city: str`）或者 Pydantic 模型（`args_schema`），自动推导出该工具需要哪些参数、参数是什么类型。

- **转化为 JSON Schema：** LangChain 会将上述提取的信息，自动转换成标准的 **JSON Schema** 格式。这是一种机器和模型都能完美理解的结构化数据格式。

- **注入系统提示词：** 在你调用 `agent.invoke()` 时，LangChain 会在底层的系统提示词（System Prompt）中，把这些工具的 JSON Schema 一并发送给 LLM。这就相当于在对话开始前，先给 LLM 发了一本“可用工具手册”，告诉它：“你现在拥有这些超能力，它们的用法和参数要求如下……”

### **2. LangChain 是怎么判断 LLM 想调用 Tools 的？**

这主要得益于现代大模型（如 DeepSeek、GPT-4 等）原生支持的 **结构化输出（Function Calling / Tool Calling）** 能力。

- **模型的原生决策：** 当 LLM 接收到包含“工具手册”的提示词和用户的提问（比如“今天天气怎么样？”）后，它会进行自我推理。如果它发现自己无法直接回答（比如不知道实时天气），它就会决定调用工具。

- 返回结构化指令：

   此时，LLM 

  不会

  直接生成一段“我要调用天气工具”的普通文本，而是会返回一个

  特定的结构化对象（通常是 JSON 格式）

  。这个对象里明确包含了：

  - `name`：它想调用的工具名称（例如 `"get_weather_for_location"`）。
  - `arguments`：它根据用户提问提取出的具体参数（例如 `{"city": "广州"}`）。

- LangChain 的拦截与执行：

   LangChain 的 Agent 框架在收到 LLM 的响应后，会先检查返回的消息里是否包含 

  ```
  tool_calls
  ```

  （工具调用指令）。

  - 如果**包含**：LangChain 就会拦截这条消息，**不让它直接输出给用户**，而是根据 `name` 找到对应的 Python 函数，把 `arguments` 传进去执行，拿到真实的运行结果（比如“广州今天晴，28度”）。
  - 如果**不包含**：说明 LLM 认为可以直接回答，LangChain 就会直接把 LLM 的文本回复返回给用户。

### **整个闭环流程是这样的：**

1. **准备阶段**：LangChain 把你的 `get_weather_for_location` 等工具打包成 JSON 格式，悄悄塞进发给 LLM 的提示词里。
2. **思考阶段**：LLM 看到用户问天气，翻了一下提示词里的“工具手册”，发现自己有天气工具，于是决定调用，并返回一个包含工具名和参数的结构化指令。
3. **执行阶段**：LangChain 捕获到这个指令，在后台默默执行了你的 Python 函数，拿到真实天气数据。
4. **总结阶段**：LangChain 把工具返回的真实数据（“广州今天晴”）再次发给 LLM，并告诉它：“这是你刚才调用的工具返回的结果”。LLM 结合这个结果，最终生成了带有双关语的完美回复。

正是因为这套机制，你不需要手动写复杂的正则表达式去判断 LLM 的意图，LangChain 和现代 LLM 已经帮你把“意图识别 -> 参数提取 -> 函数执行”的全流程自动化了。

