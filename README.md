# 文档问答助手

## 项目简介

本项目是一个基于 RAG 的文档问答系统：上传文档 → 提问 → 基于文档回答。系统会先从文档片段中检索相关内容，再根据相关性阈值选择“基于文档回答”或“模型自由回答”。

## 功能清单

- 支持上传单个 DOCX、TXT 或 PDF 文件，自动提取文本、切分片段并建立 Chroma 向量库。
- 按相似度阈值和 top-k 检索；达到阈值时使用文档上下文，未达到时不传入文档上下文。
- 支持 DeepSeek 流式回答，保留聊天记录并标注回答模式。
- 展示参考片段、相似度分数、来源文件名和前三个分块预览。
- 支持调节相似度阈值和 top-k、导出 Markdown 对话、清空对话及清除文档。

## 技术栈

- **LangChain LCEL**：用 `prompt | llm | parser` 组合 Runnable 管道，并通过 `stream()` 生成流式回答。
- **Chroma**：保存本次会话的文档向量，并按 cosine 距离检索片段。
- **BGE 本地嵌入**：使用 `BAAI/bge-small-zh-v1.5` 在本机生成文本向量，不调用在线嵌入 API。
- **DeepSeek**：通过 `ChatOpenAI` 兼容接口调用 `deepseek-flash`。
- **Streamlit**：使用纯 Python 实现左右分栏、上传控件和聊天界面。

## 文件结构

```text
F:\项目\
├── app.py                      # 主程序：界面、文档处理、检索与问答
├── requirements.txt            # 应用直接依赖及对应版本
├── README.md                   # 项目说明与运行指南
├── .streamlit\
│   └── config.toml             # 浅色主题、主色调和圆角
├── 演示文档.docx                # 测试用文档：三种格式任选其一
├── 演示文档.txt
└── 演示文档.pdf
```

## 环境与运行

### 安装依赖

在项目根目录打开终端，执行：

```powershell
F:\项目\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 设置 DeepSeek API Key

在 DeepSeek 开放平台创建 API Key，然后在 PyCharm 的 **Run → Edit Configurations → Environment variables** 中添加 `DEEPSEEK_API_KEY=你的密钥`。不要把真实密钥写进 `app.py` 或提交到代码仓库。

### 设置本地 BGE 模型

程序会优先读取环境变量 `BGE_MODEL_PATH`；未设置时，会检查 `F:\models\bge-small-zh-v1.5`，该目录存在就直接从本地加载。只有本地目录不存在时，才回退到 Hugging Face 模型名。也可以在 PyCharm 运行配置中显式设置 `BGE_MODEL_PATH=F:\models\bge-small-zh-v1.5`。

### 启动应用

在 `F:\项目` 目录执行下面这条命令：

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

主题配置文件应放在项目根目录下的 `.streamlit\config.toml`，不要放到 `.idea` 或虚拟环境目录中。

## 常见问题

### 为什么直接执行 `streamlit run app.py` 可能报“找不到模块”？

这条命令可能调用系统 Python 对应的 Streamlit，而系统 Python 没有本项目的依赖。请从 `F:\项目` 目录使用 `F:\项目\.venv\Scripts\streamlit.exe run app.py`，确保启动器和依赖属于同一个虚拟环境。

### 首次加载模型为什么比较慢？

首次加载需要读取本地模型文件并初始化 PyTorch 与嵌入模型。如果本地模型目录不存在，程序会回退到 Hugging Face 模型名，首次运行可能还要下载模型；网络不便时可设置 `HF_ENDPOINT=https://hf-mirror.com`，或把 ModelScope 下载的模型目录设置为 `BGE_MODEL_PATH`。

### TXT 中的中文显示异常怎么办？

程序会尝试自动识别 TXT 编码。如果内容仍然乱码，请将文件另存为 UTF-8 编码后重新上传。
