import streamlit as st
import os, json, re, unicodedata
from typing import List, Dict, Tuple
from collections import defaultdict

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# ======================
# 기본 설정
# ======================
st.set_page_config(page_title="불교 챗봇", layout="wide")
st.title("불교학술원")

GOOGLE_API_KEY = "AIzaSyA4H_SJa3vXNSdPge8PujTHuFrkOAjm2mw"
genai.configure(api_key=GOOGLE_API_KEY)

LLM_MODEL = "models/gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# 경로
JSON_PATH_MAIN      = r"C:\Users\User\Desktop\budda/bdc_ids3.json"
JSON_PATH_GLOSSARY  = r"C:\Users\User\Desktop\budda/bdword_structured.json"
AUX_DOC_PATH        = r"C:\Users\User\Desktop\budda/meta_sheet/문헌정보.json"
AUX_PERSON_PATH     = r"C:\Users\User\Desktop\budda/meta_sheet/인물정보.json"

# ======================
# 유틸
# ======================
def _safe_load_json(path: str):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s).lower().replace("\u200b","")
    # 공백/약한 기호 제거
    return re.sub(r"\s+","", s)

# 조사·접미 간단 제거(무한 과제축소 방지 위해 보수적으로)
_KO_SUFFIXES = [
    "에대해","에대한","에관한","에관해","이란","란","이라","라",
    "으로","로부터","로써","에서","에게","부터","까지","처럼","보다",
    "만","도","은","는","이","가","을","를","의","과","와","로","에","께","야","라도","마저","조차","뿐"
]
def _strip_suffix(token: str) -> str:
    t = token
    changed = True
    while changed and len(t) >= 3:
        changed = False
        for suf in _KO_SUFFIXES:
            if t.endswith(suf) and len(t) > len(suf)+1:
                t = t[: -len(suf)]
                changed = True
                break
    return t

def extract_terms(q: str) -> List[str]:
    base = re.findall(r"[가-힣一-龥A-Za-z0-9]{2,}", q or "")
    nq = _norm(q)
    if 3 <= len(nq) <= 60:
        base.append(nq)
    out, seen = [], set()
    for t in base:
        n = _norm(t)
        if not n: continue
        cand = {n, _strip_suffix(n)}
        for c in cand:
            if c and c not in seen:
                seen.add(c); out.append(c)
    return out

# ======================
# 문헌/인물 메타
# ======================
@st.cache_resource
def load_aux_maps() -> Tuple[Dict[str, dict], Dict[str, dict]]:
    doc_rows = _safe_load_json(AUX_DOC_PATH)
    person_rows = _safe_load_json(AUX_PERSON_PATH)
    doc_map = {str(r.get("ID")).strip(): r for r in doc_rows if r.get("ID")}
    person_map = {str(r.get("ID")).strip(): r for r in person_rows if r.get("ID")}
    return doc_map, person_map

def _intlike(x):
    try: return int(float(str(x)))
    except: return None

def brief_person(row: dict) -> str:
    if not row: return ""
    name = row.get("통칭") or row.get("이름") or ""
    ho = row.get("법호") or ""
    b = _intlike(row.get("생년")); d = _intlike(row.get("몰년"))
    years = f"({b if b is not None else '?'}–{d if d is not None else '?'})" if (row.get("생년") or row.get("몰년")) else ""
    dyn = row.get("시대(왕조)") or ""
    return f"{name}{('·'+ho) if ho else ''} {years} {dyn}".strip()

def brief_doc(row: dict) -> str:
    if not row: return ""
    ko = row.get("한글") or row.get("제목") or ""
    hanja = row.get("한자") or ""
    return f"{ko}{('['+hanja+']') if hanja else ''}".strip()

# ======================
# 코퍼스/인덱스
# ======================
@st.cache_resource(show_spinner=True)
def load_corpus_and_index():
    doc_map, person_map = load_aux_maps()
    rows = _safe_load_json(JSON_PATH_MAIN)

    docs: List[Document] = []
    for r in rows:
        title = r.get("title","")
        kws = r.get("keyword") if isinstance(r.get("keyword"), list) else ([r.get("keyword")] if r.get("keyword") else [])
        meta_ids = r.get("meta_ids") or []

        meta_tags = []
        for mid in meta_ids:
            mid = str(mid).strip()
            if mid.startswith("P") and mid in person_map:
                meta_tags.append(brief_person(person_map[mid]))
            elif mid.startswith("D") and mid in doc_map:
                meta_tags.append(brief_doc(doc_map[mid]))

        text = "\n".join([
            f"제목: {title}",
            f"내용: {r.get('content','')}",
            f"요약: {r.get('summary','')}",
            f"키워드: {', '.join([k for k in kws if k])}",
            f"메타: {', '.join([m for m in meta_tags if m])}",
            f"URL: {r.get('url','')}"
        ]).strip()

        docs.append(Document(
            page_content=text,
            metadata={
                "title": title,
                "url": r.get("url",""),
                "meta_ids": meta_ids,
                "keywords": kws
            }
        ))

    # 너무 길면만 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    final_docs: List[Document] = []
    for d in docs:
        if len(d.page_content) > 2600:
            for i, part in enumerate(splitter.split_text(d.page_content), start=1):
                md = dict(d.metadata); md["part"]=i
                final_docs.append(Document(page_content=part, metadata=md))
        else:
            final_docs.append(d)

    emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vs = FAISS.from_documents(final_docs, emb)

    norm_contents = [ _norm(d.page_content) for d in final_docs ]
    norm_titles   = [ _norm(d.metadata.get("title","")) for d in final_docs ]
    norm_keywords = [ [_norm(k) for k in (d.metadata.get("keywords") or [])] for d in final_docs ]

    title_index = defaultdict(list)
    for i, d in enumerate(final_docs):
        title_index[d.metadata.get("title","")].append(i)  # 인덱스 저장

    glossary_rows = _safe_load_json(JSON_PATH_GLOSSARY)

    return final_docs, vs, norm_contents, norm_titles, norm_keywords, title_index, glossary_rows, doc_map, person_map

# ======================
# 검색: 1순위(직접매칭) + 2순위(임베딩)
# ======================
def retrieve(query: str,
             docs: List[Document],
             vs: FAISS,
             norm_contents: List[str],
             norm_titles: List[str],
             norm_keywords: List[List[str]],
             title_index,
             k_total: int = 10) -> Tuple[List[int], List[int]]:
    """
    return: (primary_idxs, secondary_idxs)
      - primary: 질의 토큰이 본문/제목/키워드에 직접 포함된 문서(정확 매칭) + 같은 제목 보강
      - secondary: 임베딩 상위에서 비중복 보충
    """
    tokens = extract_terms(query)
    if not tokens:
        # 임베딩만
        vec = vs.as_retriever(search_kwargs={"k": k_total}).get_relevant_documents(query)
        ids = []
        for v in vec:
            # 위치 찾기
            for i, d in enumerate(docs):
                if d is v:
                    ids.append(i); break
        return ids, []

    # 1) 직접 매칭 스코어
    scores = defaultdict(float)
    for i, (ct, tt, kws) in enumerate(zip(norm_contents, norm_titles, norm_keywords)):
        hit = False
        for t in tokens:
            if not t: continue
            if t in ct:         scores[i] += 6.0; hit = True
            if t in tt:         scores[i] += 7.0; hit = True
            if any(t in kw for kw in kws): scores[i] += 5.0; hit = True
        if hit:
            # 같은 제목 보강(제목 내 다른 파트 1개만)
            title = docs[i].metadata.get("title","")
            for j in title_index.get(title, [])[:2]:
                if j != i:
                    scores[j] += 2.0  # 약한 보강

    primary = [idx for idx,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    primary = primary[:min(k_total, 8)]  # 과다 방지

    # 2) 임베딩 보충
    sec = []
    try:
        vec_docs = vs.as_retriever(search_kwargs={"k": k_total}).get_relevant_documents(query)
        for vd in vec_docs:
            for i, d in enumerate(docs):
                if d is vd and i not in primary and i not in sec:
                    sec.append(i); break
    except Exception:
        pass

    # 총량 제한
    secondary_room = max(0, k_total - len(primary))
    secondary = sec[:secondary_room]
    return primary, secondary

# ======================
# 보강: meta_ids / glossary
# ======================
def meta_briefs_for(doc: Document, doc_map: Dict[str,dict], person_map: Dict[str,dict], max_n=3) -> List[str]:
    lines = []
    for mid in (doc.metadata.get("meta_ids") or []):
        mid = str(mid).strip()
        if mid.startswith("P") and mid in person_map:
            b = brief_person(person_map[mid])
            if b: lines.append(f"인물: {b}")
        elif mid.startswith("D") and mid in doc_map:
            b = brief_doc(doc_map[mid])
            if b: lines.append(f"문헌: {b}")
        if len(lines) >= max_n: break
    # 중복 제거
    uniq, seen = [], set()
    for x in lines:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq[:max_n]

def glossary_snips(glossary_rows: List[dict], query: str, picked_text: str, max_n=3) -> List[str]:
    tokens = extract_terms(query) + list(set(re.findall(r"[가-힣一-龥A-Za-z0-9]{2,}", picked_text)))
    tokens = list({ _norm(t) for t in tokens if t })
    base = _norm(picked_text)
    out = []
    for g in glossary_rows:
        term = g.get("word") or g.get("term") or g.get("용어") or g.get("title") or ""
        if not term: continue
        nt = _norm(term)
        if nt in tokens or nt in base:
            summ = g.get("summary") or g.get("요약") or ""
            expl = g.get("content") or g.get("설명") or ""
            short = summ if summ else (expl[:140] + ("…" if len(expl) > 140 else ""))
            if short:
                out.append(f"{term}: {short}")
        if len(out) >= max_n: break
    return out

# ======================
# LLM 프롬프트 (출처 지시는 제거: 중복 방지)
# ======================
ANSWER_PROMPT = PromptTemplate.from_template("""
너는 불교 전문가야. 아래 문서를 참고해 질문에 대해 친절하고 자세하게 설명해줘. 가능한 한 풍부한 내용을 포함해 알려줘.
특히 인물이나 역사, 배경지식 같은 정보도 충분히 활용하여 답변에 포함시켜.
아래 컨텍스트만 근거로, 질문에 대한 자연스러운 서술 답변을 작성하라.
- 한글, 단일 서술(불릿/섹션 금지), 중복·군더더기 제거, 사실만.


[컨텍스트]
{context}

[질문]
{question}
""")

# ======================
# 로딩
# ======================
with st.spinner("인덱싱 준비 중..."):
    docs, vs, norm_contents, norm_titles, norm_keywords, title_index, glossary_rows, doc_map, person_map = load_corpus_and_index()

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)

# ======================
# UI
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("무엇이든 물어보세요 ")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                primary_idx, secondary_idx = retrieve(
                    q, docs, vs, norm_contents, norm_titles, norm_keywords, title_index, k_total=10
                )

                # 컨텍스트 구성: 1순위 먼저, 부족하면 2순위 보충
                picked_idx = primary_idx + secondary_idx
                if not picked_idx:
                    st.error("관련 문서를 찾지 못했습니다.")
                    st.session_state.messages.append({"role":"assistant","content":"관련 문서를 찾지 못했습니다."})
                else:
                    context_parts = []
                    source_titles = []
                    for i in picked_idx[:8]:
                        d = docs[i]
                        meta_lines = meta_briefs_for(d, doc_map, person_map, max_n=3)
                        meta_block = (" [관련 메타] " + " / ".join(meta_lines)) if meta_lines else ""
                        if d.metadata.get("title") and d.metadata.get("title") not in source_titles:
                            # 근거는 1순위(정확매칭)에서만 우선 취함
                            pass
                        context_parts.append(d.page_content + meta_block)

                    # 용어 보강은 1순위 본문 기준
                    primary_text = "\n".join([docs[i].page_content for i in primary_idx]) if primary_idx else ""
                    gloss = glossary_snips(glossary_rows, q, primary_text, max_n=3)
                    if gloss:
                        context_parts.append("[용어 보강] " + " | ".join(gloss))

                    # LLM 호출
                    ctx = "\n\n---\n\n".join(context_parts).strip()
                    prompt = ANSWER_PROMPT.format(context=ctx, question=q)
                    ans = llm.invoke(prompt).content.strip()

                    # 근거: 1순위 문서들 제목만(잡음 제거)
                    ref_titles = []
                    for i in primary_idx:
                        t = docs[i].metadata.get("title","")
                        if t and t not in ref_titles:
                            ref_titles.append(t)
                    # 1순위가 하나도 없으면 2순위로 대체
                    if not ref_titles:
                        for i in secondary_idx:
                            t = docs[i].metadata.get("title","")
                            if t and t not in ref_titles:
                                ref_titles.append(t)

                    if ref_titles:
                        ans = ans + "\n\n주요 근거: " + ", ".join(ref_titles[:5])

                    st.markdown(ans)
                    st.session_state.messages.append({"role":"assistant","content":ans})

            except Exception as e:
                st.error(f"❌ 오류: {e}")
