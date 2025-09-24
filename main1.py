import streamlit as st
import json
import re
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# --- 설정 ---
GOOGLE_API_KEY =  "AIzaSyCx-2QgdGbWxbx8RcNfz7vMPQw5itWBQuo"
genai.configure(api_key=GOOGLE_API_KEY)

LLM_MODEL = "models/gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
JSON_PATH = r"C:\Users\User\Desktop\budda\bdc_keywords_structured.json"

# --- Prompt 템플릿 개선 ---
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


# --- 문서 로딩 및 분할 ---
@st.cache_resource
def load_and_split_documents(json_path):
    raw_docs = []
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for entry in json_data:
        content = f"제목: {entry.get('title')}\n내용: {entry.get('content')}\n요약: {entry.get('summary')}\n키워드: {', '.join(entry.get('keyword', []))}\nURL: {entry.get('url')}"
        raw_docs.append(Document(
            page_content=content,
            metadata={
                "title": entry.get("title"),
                "url": entry.get("url"),
                "chunk": entry.get("chunk")
            }
        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    return splitter.split_documents(raw_docs)


# --- 벡터 스토어 생성 ---
@st.cache_resource
def create_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    return FAISS.from_documents(_docs, embeddings), embeddings


# --- 키워드 추출 개선 ---
def extract_keyword_from_query(query):
    cleaned = re.sub(r'(에 대해|이란|란|을|를|에|설명|알려|해줘|줘|주세요).*', '', query)
    return cleaned.strip()


# --- 문서 검색 ---
def find_docs_by_keyword_title(docs, keyword):
    matched_titles = {doc.metadata['title'] for doc in docs if keyword.lower() in doc.page_content.lower()}
    return [doc for doc in docs if doc.metadata['title'] in matched_titles]


# --- 질문 임베딩 ---
def embed_query(llm_embeddings, query_text):
    return llm_embeddings.embed_query(query_text)


# --- Streamlit UI ---
st.set_page_config(page_title="불교 챗봇", layout="wide")
st.title("불교 챗봇")
st.markdown("불교 문서 기반의 키워드 검색 + RAG")

with st.spinner("문서 로딩 중..."):
    documents = load_and_split_documents(JSON_PATH)
vector_store, embeddings = create_vector_store(documents)

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 대화 UI ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("질문 입력..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                keyword = extract_keyword_from_query(prompt)
                matched_docs = find_docs_by_keyword_title(documents, keyword)

                if not matched_docs:
                    matched_docs = vector_store.similarity_search(prompt, k=5)

                query_embedding = embed_query(embeddings, prompt)
                subset_store = FAISS.from_documents(matched_docs, embeddings)
                retrieved_docs = subset_store.similarity_search_by_vector(query_embedding, k=5)

                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                response = llm.invoke(qa_template.format(context=context, question=prompt))
                answer = response.content

                sources = "\n\n---\n#### 📚 참고 문서:\n"
                sources += "\n".join(f"- [{doc.metadata['title']}]({doc.metadata['url']})" for doc in retrieved_docs)

                st.markdown(answer + sources)
                st.session_state.messages.append({"role": "assistant", "content": answer + sources})

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
