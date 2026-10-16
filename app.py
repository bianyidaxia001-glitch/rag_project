# -*- coding: utf-8 -*-
"""基于 RAG 的文档问答助手。

这个文件把界面、文档处理和问答流程放在一起，方便初学者按执行顺序阅读。
RAG 的意思是：先从自己的文档中检索资料，再把资料交给大模型回答。
"""

# 第 1 部分：导入标准库和项目使用的第三方库。
# 标准库负责文件、计时和哈希；第三方库分别负责界面、加载文档、向量检索和大模型。 
import hashlib
import html
import os
import tempfile
import textwrap
import time
import uuid
from datetime import datetime

import httpx
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 第 2 部分：设置页面标题、布局和轻量 CSS。
# CSS 只影响外观；删掉这段 CSS 后，上传、检索和问答功能仍然正常。
st.set_page_config(
    page_title="文档问答助手",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* 页面使用浅灰底色，让白色内容卡片更清楚。 */
    body { background: #f6f7fb; }
    /* 顶部标题区留出空间，并使用浅紫色渐变作点缀。 */
    .hero-banner { padding: 1.2rem 1.5rem; border-radius: 20px; background: linear-gradient(120deg, #e6e8ff 0%, #f3f4ff 45%, #ffffff 100%); background-size: 200% 200%; animation: heroFlow 16s ease-in-out infinite; margin-bottom: 1rem; }
    /* 渐变背景缓慢左右流动（16 秒一个来回），像轻微的呼吸。 */
    @keyframes heroFlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    /* 标题区主标题使用深色和较紧凑的行距。 */
    .hero-banner h1 { color: #20244a; margin: 0 0 .35rem 0; font-size: 2rem; }
    /* 标题区说明文字降低对比度，突出主标题。 */
    .hero-banner p { color: #626781; margin: 0; }
    /* 标题区的格式胶囊：一眼展示支持 DOCX、TXT、PDF 三种格式。 */
    .hero-tags { margin-top: .65rem; display: flex; gap: .45rem; }
    .hero-tag { background: #ffffff; border: 1px solid #dfe2ff; color: #4d53a8; font-size: .78rem; font-weight: 600; padding: .12rem .65rem; border-radius: 999px; }
    /* 上传区卡片使用圆角和柔和阴影。 */
    .st-key-upload_card { border-radius: 18px; box-shadow: 0 8px 24px rgba(36, 42, 92, .06); }
    /* 文档指标卡片沿用同一套圆角和阴影。 */
    .st-key-doc_info_card { border-radius: 18px; box-shadow: 0 8px 24px rgba(36, 42, 92, .06); }
    /* 参数区卡片留出舒适的内边距。 */
    .st-key-parameter_card { border-radius: 18px; }
    /* 聊天消息之间增加间距，并让每条消息呈现为白色圆角卡片。 */
    .st-key-chat_area [data-testid="stChatMessage"] { margin-bottom: .75rem; background: rgba(255, 255, 255, .9); border: 1px solid #edeefa; border-radius: 16px; padding: .55rem .95rem; animation: messageIn .35s ease-out both; }
    /* 新消息出场动效：从下方 8 像素淡入上浮。 */
    @keyframes messageIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    /* 按钮使用圆角，和主题配置保持一致。 */
    button { border-radius: 12px !important; transition: transform .15s ease, box-shadow .15s ease; }
    /* 鼠标悬停时轻微抬起按钮，提供交互反馈。 */
    button:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(99, 102, 241, .14); }
    /* 按下时回落，模拟按下去的物理反馈。 */
    button:active { transform: translateY(0); box-shadow: 0 2px 6px rgba(99, 102, 241, .12); }
    /* 指标数字加粗加深，演示读数时更醒目。 */
    [data-testid="stMetricValue"] { font-weight: 700; color: #3a3d68; }
    /* 提示条与展开面板统一圆角，和整体卡片风格一致。 */
    [data-testid="stAlert"], [data-testid="stExpander"] { border-radius: 12px; }
    /* 滚动条纤细化：浅紫圆角，和主题色呼应。 */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: #d8daf2; border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: #c3c6ee; }
    /* 回答模式徽章：绿色=基于文档回答，灰色=自由回答，橙色=其他提示。 */
    .mode-pill { display: inline-block; padding: .14rem .62rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }
    .mode-ok { background: #e7f7ee; color: #1c7c4d; }
    .mode-free { background: #f1f2f6; color: #6a6f89; }
    .mode-warn { background: #fff3e0; color: #b26a00; }
    /* 参考片段卡片：把来源、分数、进度条和正文收进一张小卡。 */
    .ref-card { background: #fafbff; border: 1px solid #e8eaff; border-radius: 14px; padding: .7rem .9rem; margin-bottom: .6rem; }
    .ref-head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
    .ref-title { font-weight: 700; color: #2b2f55; }
    .ref-source { background: #ffffff; border: 1px solid #e8eaff; color: #6a6f89; font-size: .78rem; padding: .05rem .55rem; border-radius: 999px; }
    .ref-score { font-size: .78rem; font-weight: 700; padding: .05rem .55rem; border-radius: 999px; }
    .ref-score.score-high { background: #e7f7ee; color: #1c7c4d; }
    .ref-score.score-low { background: #fff3e0; color: #b26a00; }
    .ref-bar { height: 6px; background: #eceefb; border-radius: 999px; overflow: hidden; margin: .5rem 0; }
    .ref-bar-fill { height: 100%; border-radius: 999px; }
    .ref-bar-fill.score-high { background: #34b57a; }
    .ref-bar-fill.score-low { background: #f0a63c; }
    .ref-text { color: #3c405e; font-size: .86rem; line-height: 1.55; white-space: pre-wrap; }
    /* 示例问题按钮卡片化：文字可换行、悬停点亮。 */
    [class*="st-key-example_question_"] button { white-space: normal; text-align: left; line-height: 1.45; padding: .7rem .9rem; min-height: 4.4rem; background: #fbfbff; border: 1px solid #e3e5f8; }
    [class*="st-key-example_question_"] button:hover { background: #ffffff; border-color: #c9ccf5; }
    /* 上传区悬停时高亮描边，提示可以拖入文件。 */
    .st-key-upload_card [data-testid="stFileUploaderDropzone"] { transition: border-color .2s ease, background .2s ease; }
    .st-key-upload_card [data-testid="stFileUploaderDropzone"]:hover { border-color: #b9bdf2; background: #fafbff; }
    /* 隐藏 Streamlit 默认页脚，让页面更像完整的小型产品。 */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-banner">
      <h1>📚 文档问答助手</h1>
      <p>上传一份文档，先检索相关内容，再让模型结合资料回答。</p>
      <div class="hero-tags">
        <span class="hero-tag">DOCX</span>
        <span class="hero-tag">TXT</span>
        <span class="hero-tag">PDF</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# 第 3 部分：集中初始化会跨页面重跑保留的状态。
# Streamlit 每次点击控件都会从头运行脚本；session_state 可以记住前一次的数据。
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("document_name", "")
st.session_state.setdefault("document_char_count", 0)
st.session_state.setdefault("document_chunk_count", 0)
st.session_state.setdefault("document_preview_chunks", [])
st.session_state.setdefault("document_seconds", 0.0)
st.session_state.setdefault("document_status", "empty")
st.session_state.setdefault("document_error", "")
st.session_state.setdefault("last_attempted_upload_hash", None)
st.session_state.setdefault("upload_widget_version", 0)
st.session_state.setdefault("document_success_notice", False)
st.session_state.setdefault("document_clear_notice", False)
st.session_state.setdefault("document_clear_warning", "")


# 第 4 部分：准备本地嵌入模型和两个独立的按钮回调。
# cache_resource 让昂贵的本地模型在同一服务进程中只加载一次。
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """加载本地 BGE 模型；向量归一化后可配合 Chroma 的 cosine 距离检索。"""
    # 优先读环境变量里的本地模型路径；若环境变量缺失，自动检测 F:\models 下的本地副本；
    # 两者都没有，才回退到在线模型名（答辩现场固定走本地副本，完全零网络依赖）。
    model_name = os.getenv("BGE_MODEL_PATH") or (
        "F:/models/bge-small-zh-v1.5"
        if os.path.isdir("F:/models/bge-small-zh-v1.5")
        else "BAAI/bge-small-zh-v1.5"
    )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def clear_chat_history():
    """清空聊天记录和待提问示例，不修改文档和向量库。"""
    st.session_state["messages"] = []
    st.session_state["pending_question"] = None


def clear_document_store():
    """删除当前 Chroma 集合，并换一个上传控件键以清空文件选择框。"""
    vector_store = st.session_state.get("vector_store")
    st.session_state["document_clear_warning"] = ""

    if vector_store is not None:
        try:
            vector_store.delete_collection()
        except Exception as error:
            # 即使 Chroma 删除时报错，也移除当前会话的引用，避免继续检索旧文档。
            st.session_state["document_clear_warning"] = f"向量库清理提示：{error}"

    st.session_state["vector_store"] = None
    st.session_state["document_name"] = ""
    st.session_state["document_char_count"] = 0
    st.session_state["document_chunk_count"] = 0
    st.session_state["document_preview_chunks"] = []
    st.session_state["document_seconds"] = 0.0
    st.session_state["document_status"] = "empty"
    st.session_state["document_error"] = ""
    st.session_state["last_attempted_upload_hash"] = None
    st.session_state["upload_widget_version"] += 1
    st.session_state["document_clear_notice"] = True


def mode_badge_html(mode):
    """把回答模式转换成彩色小徽章：绿=基于文档，灰=自由回答，橙=其他提示。"""
    if mode == "基于文档回答":
        style_class = "mode-ok"
    elif mode == "模型自由回答":
        style_class = "mode-free"
    else:
        style_class = "mode-warn"
    return f'<span class="mode-pill {style_class}">{mode}</span>'


def show_answer_references(mode, references):
    """在回答下方展开显示来源；自由回答时明确说明没有采用文档片段。"""
    with st.expander("查看参考片段"):
        if mode == "基于文档回答":
            if not references:
                st.info("本次没有可展示的参考片段。")
            for index, item in enumerate(references, start=1):
                score = item["score"]
                # 相似度 0.6 及以上显示绿色，其余显示橙色，方便一眼区分强弱。
                score_class = "score-high" if score >= 0.6 else "score-low"
                # 来源和正文都做转义，防止文档里的特殊符号破坏卡片结构。
                source_text = html.escape(str(item["source"]))
                content = item["content"]
                short_content = content[:200] + ("……" if len(content) > 200 else "")
                text_text = html.escape(short_content)
                # textwrap.dedent 去掉字符串缩进，让 HTML 从行首开始（Markdown 才能识别成网页元素）。
                card_html = textwrap.dedent(
                    f"""
                    <div class="ref-card">
                      <div class="ref-head">
                        <span class="ref-title">片段 {index}</span>
                        <span class="ref-source">来源文件：{source_text}</span>
                        <span class="ref-score {score_class}">相似度：{score:.3f}</span>
                      </div>
                      <div class="ref-bar"><div class="ref-bar-fill {score_class}" style="width: {score * 100:.1f}%"></div></div>
                      <div class="ref-text">{text_text}</div>
                    </div>
                    """
                ).strip()
                st.markdown(card_html, unsafe_allow_html=True)
        elif mode == "模型自由回答":
            st.info("未检索到相关片段，以下回答来自模型自身知识。")
        elif mode == "需要上传文档":
            st.info("请先在左侧上传 DOCX、TXT 或 PDF 文档，再开始提问。")
        elif mode == "需要配置 API Key":
            st.info("请设置环境变量 DEEPSEEK_API_KEY，然后重新提问。")
        else:
            st.info("本次未生成可用的参考片段。")


# 第 5 部分：建立左右分栏，并在左边上传、读取和处理文档。
# 上传后用文件内容的 SHA-256 指纹判断是否处理过，避免普通页面重跑再次向量化。
left_column, right_column = st.columns([0.36, 0.64], gap="large")

with left_column:
    st.subheader("📄 文档知识库")
    with st.container(border=True, key="upload_card"):
        uploader_key = f"document_uploader_{st.session_state['upload_widget_version']}"
        uploaded_file = st.file_uploader(
            "选择一份 DOCX、TXT 或 PDF 文件",
            type=["docx", "txt", "pdf"],
            accept_multiple_files=False,
            key=uploader_key,
        )

        if uploaded_file is None and st.session_state["document_status"] == "empty":
            st.info("先上传文档，系统会自动完成文本提取、分块和向量化。")

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            # 文件名也参与指纹；同一内容改名后，来源显示仍会更新。
            upload_hash = hashlib.sha256(
                uploaded_file.name.encode("utf-8") + b"\0" + file_bytes
            ).hexdigest()

            if upload_hash != st.session_state["last_attempted_upload_hash"]:
                st.session_state["document_status"] = "processing"
                st.session_state["document_error"] = ""
                start_time = time.perf_counter()
                temporary_path = None

                # 新上传的文档会替换旧文档，避免检索结果混入上一份资料。
                old_store = st.session_state.get("vector_store")
                if old_store is not None:
                    try:
                        old_store.delete_collection()
                    except Exception:
                        pass
                st.session_state["vector_store"] = None

                try:
                    with st.spinner("正在提取文本、切分内容并建立向量库，请稍候……"):
                        extension = os.path.splitext(uploaded_file.name)[1].lower()

                        # LangChain 的文件加载器接收文件路径，因此先把上传内容临时写入磁盘。
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=extension, delete=False
                        ) as temporary_file:
                            temporary_file.write(file_bytes)
                            temporary_path = temporary_file.name

                        # 根据扩展名选择课程要求的 DOCX、TXT 或 PDF 加载器。
                        if extension == ".docx":
                            loader = Docx2txtLoader(temporary_path)
                        elif extension == ".txt":
                            loader = TextLoader(
                                temporary_path,
                                encoding="utf-8",
                                autodetect_encoding=True,
                            )
                        elif extension == ".pdf":
                            loader = PyPDFLoader(temporary_path)
                        else:
                            raise ValueError("只支持 DOCX、TXT 和 PDF 文件。")

                        loaded_documents = loader.load()
                        full_text = "\n".join(
                            document.page_content for document in loaded_documents
                        ).strip()
                        if not full_text:
                            raise ValueError("没有从文件中提取到文字，请检查文件内容。")

                        # 将加载器临时路径替换成用户上传时看到的文件名，便于展示来源。
                        for document in loaded_documents:
                            document.metadata["source"] = uploaded_file.name

                        # 两个切分参数按项目要求固定，不放到界面上调整。
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                        )
                        chunks = splitter.split_documents(loaded_documents)
                        if not chunks:
                            raise ValueError("文档切分后没有可用片段。")

                        # 只保留前三个片段的短文本，用于界面预览，避免把整份文档复制进会话状态。
                        preview_chunks = [
                            chunk.page_content[:200]
                            + ("……" if len(chunk.page_content) > 200 else "")
                            for chunk in chunks[:3]
                        ]

                        # 每次建库使用唯一集合名，防止不同会话的临时数据相互覆盖。
                        # cosine 表示使用余弦距离；归一化在上方嵌入模型设置中完成。
                        vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=get_embedding_model(),
                            collection_name=f"document_{uuid.uuid4().hex}",
                            collection_metadata={"hnsw:space": "cosine"},
                            # cosine 距离为 1 - 余弦相似度；夹到 0 到 1 便于阈值和进度条使用。
                            relevance_score_fn=lambda distance: max(
                                0.0, min(1.0, 1.0 - distance)
                            ),
                        )

                    st.session_state["vector_store"] = vector_store
                    st.session_state["document_name"] = uploaded_file.name
                    st.session_state["document_char_count"] = len(full_text)
                    st.session_state["document_chunk_count"] = len(chunks)
                    st.session_state["document_preview_chunks"] = preview_chunks
                    st.session_state["document_seconds"] = time.perf_counter() - start_time
                    st.session_state["document_status"] = "ready"
                    st.session_state["document_success_notice"] = True
                except Exception as error:
                    st.session_state["document_status"] = "error"
                    st.session_state["document_error"] = str(error)
                finally:
                    # 临时副本只用于加载器读取，处理结束后立即删除。
                    if temporary_path and os.path.exists(temporary_path):
                        os.remove(temporary_path)
                    # 失败时也记录尝试过的文件，避免每次滑块变化都重复处理。
                    st.session_state["last_attempted_upload_hash"] = upload_hash

        if st.session_state["document_status"] == "ready":
            st.success("文档处理完成，可以开始提问。")
        elif st.session_state["document_status"] == "processing":
            st.info("文档正在处理中。")
        elif st.session_state["document_status"] == "error":
            st.error(f"文档处理失败：{st.session_state['document_error']}")

    # 处理成功后展示本次文件名、字符数、片段数和耗时。
    if st.session_state["document_status"] == "ready":
        with st.container(border=True, key="doc_info_card"):
            st.markdown("**📊 文档信息**")
            st.caption(f"文件名：{st.session_state['document_name']}")
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "文本字符数", f"{st.session_state['document_char_count']:,}"
            )
            metric_columns[1].metric(
                "分块数量", st.session_state["document_chunk_count"]
            )
            metric_columns[2].metric(
                "处理耗时（秒）", f"{st.session_state['document_seconds']:.2f}"
            )

        # 分块预览只展示片段总数和前三段的短文本，方便答辩时说明切分结果。
        with st.expander("查看分块结果"):
            st.caption(f"共 {st.session_state['document_chunk_count']} 个片段")
            for index, preview_text in enumerate(
                st.session_state["document_preview_chunks"], start=1
            ):
                st.markdown(f"**片段 {index}**")
                st.write(preview_text)

    # 两个滑块控制下一次提问的检索方式，不会重新处理已经上传的文件。
    with st.container(border=True, key="parameter_card"):
        st.markdown("**⚙️ 检索参数**")
        similarity_threshold = st.slider(
            "相似度阈值",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            key="similarity_threshold",
            help="最高相似度高于阈值时使用文档回答，否则使用模型自由回答。",
        )
        top_k = st.slider(
            "检索片段数量（top-k）",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key="top_k",
            help="每次提问最多检索多少个文档片段。",
        )

    if st.button("🗑️ 清除文档", key="clear_documents_button", width="stretch"):
        clear_document_store()
        st.rerun()

    if st.session_state["document_clear_warning"]:
        st.warning(st.session_state["document_clear_warning"])
    if st.session_state["document_clear_notice"]:
        st.toast("已清除文档和向量库。")
        st.session_state["document_clear_notice"] = False
    if st.session_state["document_success_notice"]:
        st.toast("文档已加入知识库。")
        st.session_state["document_success_notice"] = False


# 第 6 部分：在右边展示聊天记录、欢迎态、示例问题和聊天输入框。
# 示例按钮只记录待提问的问题，下一次脚本运行时和普通输入走同一段问答代码。
with right_column:
    title_column, export_column, clear_column = st.columns(
        [0.54, 0.23, 0.23], vertical_alignment="center"
    )
    with title_column:
        st.subheader("💬 与文档对话")
    if st.session_state["messages"]:
        # 导出内容按问答顺序组织，并为每条回答附上实际模式标签。
        export_lines = ["# 对话记录", ""]
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                export_lines.extend(["## 问题", "", message["content"], ""])
            elif message["role"] == "assistant":
                answer_mode = message.get("mode", "模型自由回答")
                export_lines.extend(
                    [f"## 回答（{answer_mode}）", "", message["content"], ""]
                )

        with export_column:
            st.download_button(
                "导出对话",
                data="\n".join(export_lines),
                file_name=f"对话记录_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
                key="export_chat_button",
                width="stretch",
            )
    with clear_column:
        st.button(
            "清空对话",
            key="clear_chat_button",
            on_click=clear_chat_history,
            width="stretch",
        )

    if not st.session_state["messages"]:
        st.markdown("### 你好，想从文档里了解什么？")
        st.write("上传文档后，可以试试下面的问题：")
        example_questions = [
            "请总结这份文档的主要内容。",
            "文档中提到了哪些关键数据或结论？",
            "这份文档适用于什么场景？",
        ]
        example_columns = st.columns(3)
        for index, example_question in enumerate(example_questions):
            if example_columns[index].button(
                example_question,
                key=f"example_question_{index}",
                width="stretch",
            ):
                st.session_state["pending_question"] = example_question
                st.rerun()

    # 历史消息逐条重画；assistant 消息下方同时恢复回答模式和参考片段。
    # 自定义聊天头像：提问方 🙋、回答方 📚，让对话更醒目。
    with st.container(key="chat_area"):
        for message in st.session_state["messages"]:
            with st.chat_message(
                message["role"],
                avatar="🙋" if message["role"] == "user" else "📚",
            ):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    st.markdown(
                        mode_badge_html(message.get("mode", "模型自由回答")),
                        unsafe_allow_html=True,
                    )
                    if message.get("error_detail"):
                        st.caption(f"错误详情：{message['error_detail']}")
                    show_answer_references(
                        message.get("mode", "模型自由回答"),
                        message.get("references", []),
                    )

    typed_question = st.chat_input("输入你想问的问题……")
    question = typed_question or st.session_state["pending_question"]
    if question:
        question = question.strip()
        st.session_state["pending_question"] = None

        if question:
            st.session_state["messages"].append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user", avatar="🙋"):
                st.markdown(question)

            answer = ""
            mode = "模型自由回答"
            references = []
            error_detail = ""

            with st.chat_message("assistant", avatar="📚"):
                vector_store = st.session_state.get("vector_store")

                # 没有文档时给出清晰操作提示，不调用检索器或大模型。
                if vector_store is None:
                    answer = "请先在左侧上传一份 DOCX、TXT 或 PDF 文档，再开始提问。"
                    mode = "需要上传文档"
                    st.info(answer)

                # API Key 由开发者放进环境变量；缺少时说明设置方法，不发起网络请求。
                elif not os.getenv("DEEPSEEK_API_KEY", "").strip():
                    answer = "尚未设置 DEEPSEEK_API_KEY。请先在 PyCharm 运行配置中填写 API Key。"
                    mode = "需要配置 API Key"
                    st.info(answer)

                else:
                    try:
                        # 只取相关性分数；该方法分数越大表示越相关。显示前限制到 0 到 1。
                        search_results = vector_store.similarity_search_with_relevance_scores(
                            question,
                            k=top_k,
                        )
                        for document, raw_score in search_results:
                            score = max(0.0, min(1.0, float(raw_score)))
                            source_name = os.path.basename(
                                str(document.metadata.get("source", "未知文件"))
                            )
                            references.append(
                                {
                                    "content": document.page_content,
                                    "score": score,
                                    "source": source_name,
                                }
                            )

                        highest_score = max(
                            (item["score"] for item in references),
                            default=0.0,
                        )

                        # 用同一个 DeepSeek 模型分别组成文档链和自由回答链。
                        llm = ChatOpenAI(
                            model="deepseek-flash",
                            base_url="https://api.deepseek.com",
                            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
                            # 直连设置：单独给大模型调用配一个"忽略本机代理"的客户端，
                            # 避免请求被本机代理软件转发（转发到 DeepSeek 时可能握手失败）。
                            http_client=httpx.Client(
                                trust_env=False,
                                timeout=httpx.Timeout(120, connect=10),
                            ),
                        )
                        parser = StrOutputParser()

                        if references and highest_score > similarity_threshold:
                            mode = "基于文档回答"
                            context = "\n\n".join(
                                item["content"] for item in references
                            )
                            document_prompt = ChatPromptTemplate.from_messages(
                                [
                                    (
                                        "system",
                                        "你是文档问答助手。回答必须严格依据给定资料。"
                                        "如果资料没有说明，就明确说文档中没有提到，不要编造。",
                                    ),
                                    (
                                        "human",
                                        "参考资料：\n{context}\n\n用户问题：{question}",
                                    ),
                                ]
                            )
                            document_chain = document_prompt | llm | parser
                            answer = st.write_stream(
                                document_chain.stream(
                                    {"context": context, "question": question}
                                )
                            )
                        else:
                            # 未达到阈值时不把任何检索片段传入自由回答链。
                            mode = "模型自由回答"
                            free_prompt = ChatPromptTemplate.from_messages(
                                [
                                    (
                                        "system",
                                        "你是一个友好的中文问答助手。请使用自身知识回答，"
                                        "并在不确定时如实说明。",
                                    ),
                                    ("human", "{question}"),
                                ]
                            )
                            free_chain = free_prompt | llm | parser
                            answer = st.write_stream(
                                free_chain.stream({"question": question})
                            )

                    except Exception as error:
                        # 不把可能包含请求细节的异常堆栈直接展示给用户。
                        answer = "调用模型时遇到问题，请检查 API Key、网络连接和 DeepSeek 接口配置。"
                        mode = "调用失败"
                        # 只取错误首行并限制长度，展示摘要而不展示程序堆栈。
                        error_detail = (
                            str(error).splitlines()[0][:200]
                            if str(error).strip()
                            else type(error).__name__
                        )
                        st.error(answer)
                        st.caption(f"错误详情：{error_detail}")

                # 流式回答结束后再显示模式徽章和 expander，保持阅读顺序稳定。
                st.markdown(
                    mode_badge_html(mode),
                    unsafe_allow_html=True,
                )
                show_answer_references(mode, references)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "mode": mode,
                    "references": references,
                    "error_detail": error_detail,
                }
            )
