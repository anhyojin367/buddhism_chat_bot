import streamlit as st
import pandas as pd
import os
import json
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import re

# --- 설정 ---
GOOGLE_API_KEY = "AIzaSyA4H_SJa3vXNSdPge8PujTHuFrkOAjm2mw"
genai.configure(api_key=GOOGLE_API_KEY)

LLM_MODEL = "models/gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
JSON_PATH_MAIN = r"C:/Users/User/Desktop/budda/bdc_keywords_structured.json"
JSON_PATH_GLOSSARY = r"C:/Users/User/Desktop/budda//bdword_structured.json"


# --- 문서 로딩 + TextSplitter ---
@st.cache_resource
def load_and_split_documents(main_path, glossary_path):
    raw_docs = []
    for path in [main_path, glossary_path]:
        with open(path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        for entry in json_data:
            content_parts = [
                f"chunk: {entry.get('chunk', '')}",
                f"\n제목: {entry.get('title', '')}",
                f"\n내용: {entry.get('content', '')}",
                f"\n요약: {entry.get('summary', '')}",
                f"\n키워드: {', '.join(entry.get('keyword', []))}" if isinstance(entry.get("keyword"), list) else f"\n키워드: {entry.get('keyword', '')}",
                f"\nURL: {entry.get('url', '')}",
                f"\n이미지: {entry.get('image', '')}"
            ]
            full_content = "\n".join(content_parts)

            raw_docs.append(
                Document(
                    page_content=full_content,
                    metadata={
                        "title": entry.get("title", ""),
                        "url": entry.get("url", ""),
                        "chunk": entry.get("chunk", ""),
                        "source": "json"
                    }
                )
            )

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    return splitter.split_documents(raw_docs)

# --- 벡터스토어 생성 ---
@st.cache_resource(show_spinner=False)
def create_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_documents(_docs, embeddings)
    return vector_store

# --- 키워드 기반 유사 문서 검색 함수 ---
def search_documents_by_keyword(query, documents, window=2):
    keyword_candidates = re.findall(r'\w+', query)
    found_indices = set()

    for i, doc in enumerate(documents):
        for keyword in keyword_candidates:
            if keyword in doc.page_content:
                start = max(0, i - window)
                end = min(len(documents), i + window + 1)
                found_indices.update(range(start, end))

    return [documents[i] for i in sorted(found_indices)]

# --- 프롬프트 정의 ---
qa_template = PromptTemplate.from_template("""
너는 불교 전문가야. 아래 문서를 참고해 질문에 대해 친절하고 자세하게 설명해줘. 가능한 한 풍부한 내용을 포함해 알려줘.
특히 인물이나 역사, 배경지식 같은 정보도 충분히 활용하여 답변에 포함시켜.
**핵심 답변**
사용자 질문에 대한 직접적이고 명확한 답변

**상세 설명**
관련 chunk 데이터를 바탕으로 한 자세한 설명

문서 내용:
{context}

질문:
{question}

정확하고 간결한 답변:
""")

# --- Streamlit UI ---
st.set_page_config(page_title="불교 챗봇", layout="wide")
st.title("불교학술원")

# --- 문서 로딩 및 벡터스토어 생성 ---
with st.spinner("문서 불러오는 중..."):
    documents = load_and_split_documents(JSON_PATH_MAIN, JSON_PATH_GLOSSARY)
with st.spinner("벡터 스토어 생성 중..."):
    vector_store = create_vector_store(documents)

# --- LLM 초기화 ---
if "chain" not in st.session_state:
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    st.session_state.chain = chain
    st.session_state.messages = []
    st.session_state.retriever = retriever
    st.session_state.docs = documents
    st.session_state.llm = llm  # llm 객체 직접 저장

# --- 대화 기록 출력 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 사용자 질문 처리 ---
if prompt := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                keyword_docs = search_documents_by_keyword(prompt, st.session_state.docs)
                source_text = ""
                unique_refs = {}

                if keyword_docs:
                    context = "\n\n".join([doc.page_content for doc in keyword_docs])
                    prompt_text = qa_template.format(context=context, question=prompt)
                    response = st.session_state.llm.invoke(prompt_text)
                    answer = response.content

                    for doc in keyword_docs:
                        title = doc.metadata.get("title", "")
                        url = doc.metadata.get("url", "")
                        if title and title not in unique_refs:
                            unique_refs[title] = url
                else:
                    result = st.session_state.chain({"question": prompt})
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])
                    for doc in source_docs:
                        title = doc.metadata.get("title", "")
                        url = doc.metadata.get("url", "")
                        if title and title not in unique_refs:
                            unique_refs[title] = url

                if unique_refs:
                    source_text = "\n\n---\n#### 📚 참고 문서 (Top 5):\n"
                    top_refs = list(unique_refs.items())[:5]
                    for title, url in top_refs:
                        if url:
                            source_text += f"- [{title}]({url})\n"
                        else:
                            source_text += f"- {title}\n"

                st.markdown(answer + source_text)
                st.session_state.messages.append({"role": "assistant", "content": answer + source_text})

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")