import streamlit as st
import os, json, re, unicodedata
from typing import List, Dict, Tuple, Any
from collections import defaultdict

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage

# ======================
# 기본 설정
# ======================
st.set_page_config(page_title="불교 챗봇", layout="wide")
st.title("불교학술원")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # ▶︎ 운영시 환경변수 권장
genai.configure(api_key=GOOGLE_API_KEY)

LLM_MODEL = "models/gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# 현재 스크립트 파일의 디렉토리를 기준으로 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 경로
JSON_PATH_MAIN      = os.path.join(BASE_DIR, "bdc_ids3.json")
JSON_PATH_GLOSSARY  = os.path.join(BASE_DIR, "bdword_structured.json")
AUX_DOC_PATH        = os.path.join(BASE_DIR, "meta_sheet", "문헌정보.json") # meta_sheet 폴더 안의 파일
AUX_PERSON_PATH     = os.path.join(BASE_DIR, "meta_sheet", "인물정보.json") # meta_sheet 폴더 안의 파일

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

def _is_word_char(ch: str) -> bool:
    return bool(re.match(r"[A-Za-z0-9가-힣一-龥]", ch))

def _around(text: str, start: int, end: int, win: int = 24) -> str:
    s = max(0, start - win)
    e = min(len(text), end + win)
    return text[s:e]

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
# 별칭 인덱스(문자열 매칭용)
# ======================
_DOC_SUFFIXES = ("경","론","기","어록","집","도서")
_JOSA_REGEX = r"(?:으로|에게|부터|까지|처럼|보다|[이가은는을를의과와로에께야라도마저조차뿐])?"

def _alias_to_pattern(alias: str) -> str:
    # 공백/중점/하이픈 등은 느슨하게 매칭
    a = alias.strip()
    a = re.sub(r"\s+", r"\\s*", re.escape(a))
    a = a.replace(r"\·", r"(?:·|\s*)")
    return a

@st.cache_resource
def build_alias_patterns(doc_map: Dict[str,dict], person_map: Dict[str,dict]) -> List[Dict[str, Any]]:
    """
    반환: [{'id': 'P002', 'type': 'person', 'alias': '대혜 종고', 're': compiled, 'alen': 4}, ...]
    Longest-first 적용을 위해 alen 기준 내림차순 정렬
    """
    records: List[Dict[str, Any]] = []

    # 인물
    for pid, row in person_map.items():
        if not pid.startswith("P"): continue
        aliases = set()
        for k in ("통칭","이름","법호","이름_한자","법호_한자"):
            v = (row.get(k) or "").strip()
            if v: aliases.add(v)
        # 조합(과도 확장 방지: 2단계까지만)
        if row.get("통칭") and row.get("법호"):
            aliases.add(f"{row.get('통칭')} {row.get('법호')}")
        # 필터링
        final = []
        for a in aliases:
            if len(re.sub(r"\s+","", a)) < 2:  # 너무 짧으면 제외
                continue
            final.append(a)
        for a in final:
            pat = rf"(?<![A-Za-z0-9가-힣一-龥]){_alias_to_pattern(a)}{_JOSA_REGEX}(?![A-Za-z0-9가-힣一-龥])"
            records.append({"id": pid, "type": "person", "alias": a, "re": re.compile(pat), "alen": len(a)})

    # 문헌
    for did, row in doc_map.items():
        if not did.startswith("D"): continue
        aliases = set()
        for k in ("한글","제목","한자"):
            v = (row.get(k) or "").strip()
            if v: aliases.add(v)
        final = []
        for a in aliases:
            if len(re.sub(r"\s+","", a)) < 2:
                continue
            final.append(a)
        for a in final:
            # 문헌은 『…』/접미어 힌트를 활용(검증은 매칭 후 문맥에서 추가 확인)
            pat = rf"(?<![A-Za-z0-9가-힣一-龥]){_alias_to_pattern(a)}{_JOSA_REGEX}(?![A-Za-z0-9가-힣一-龥])"
            records.append({"id": did, "type": "doc", "alias": a, "re": re.compile(pat), "alen": len(a)})

    # 긴 별칭 우선
    records.sort(key=lambda x: x["alen"], reverse=True)
    return records

def _doc_match_valid(text: str, start: int, end: int, alias: str) -> bool:
    ctx = _around(text, start, end, 40)
    # 『…』 같은 표식이 근처에 있거나, 별칭이 접미어 보유
    has_bracket = ("『" in ctx and "』" in ctx) or ("「" in ctx and "」" in ctx)
    alias_has_suffix = alias.endswith(_DOC_SUFFIXES)
    return has_bracket or alias_has_suffix

def autolink_meta_ids_to_text(text: str, alias_records: List[Dict[str,Any]], max_ids: int = 12) -> Tuple[List[str], List[str]]:
    """
    텍스트에서 별칭 매칭으로 meta_ids 추출(가드레일 적용)
    반환: (ids, evidence-snippets)
    """
    taken_spans: List[Tuple[int,int]] = []
    found_ids: List[str] = []
    evidences: List[str] = []

    def conflict(s, e):
        for (xs, xe) in taken_spans:
            if not (e <= xs or xe <= s):
                return True
        return False

    for rec in alias_records:
        if len(found_ids) >= max_ids: break
        for m in rec["re"].finditer(text):
            s, e = m.span()
            if conflict(s,e):  # longest-first로 만들어서 먼저 잡힌 건 유지
                continue
            alias = rec["alias"]
            # 타입별 간단 검증
            if rec["type"] == "doc" and not _doc_match_valid(text, s, e, alias):
                continue
            # 경계 추가체크(좌우가 단어문자면 스킵)
            if s > 0 and _is_word_char(text[s-1]):
                continue
            if e < len(text) and _is_word_char(text[e:e+1]):
                continue

            found_ids.append(rec["id"])
            taken_spans.append((s,e))
            evidences.append(f"{rec['id']}:{alias} | …{_around(text, s, e, 18)}…")
            break  # 같은 별칭으로 여러 번 추가 방지
    # 중복 제거(입력 순서 유지)
    uniq, seen = [], set()
    for x in found_ids:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq, evidences[:max_ids]

# ======================
# 코퍼스/인덱스
# ======================
@st.cache_resource(show_spinner=True)
def load_corpus_and_index():
    doc_map, person_map = load_aux_maps()
    alias_records = build_alias_patterns(doc_map, person_map)

    rows = _safe_load_json(JSON_PATH_MAIN)

    docs: List[Document] = []
    for r in rows:
        title = r.get("title","")
        kws = r.get("keyword") if isinstance(r.get("keyword"), list) else ([r.get("keyword")] if r.get("keyword") else [])
        provided_meta_ids = [str(x).strip() for x in (r.get("meta_ids") or [])]

        text_raw = "\n".join([
            f"제목: {title}",
            f"내용: {r.get('content','')}",
            f"요약: {r.get('summary','')}",
            f"키워드: {', '.join([k for k in kws if k])}",
            f"URL: {r.get('url','')}"
        ]).strip()

        # ▶︎ 자동 메타 연결(문자열 매칭, 가드레일 적용)
        auto_ids, auto_evd = autolink_meta_ids_to_text(text_raw, alias_records, max_ids=12)
        merged_meta_ids = list(dict.fromkeys(provided_meta_ids + auto_ids))  # 순서 유지 dedup

        # 메타 소개 라인(간단 요약)
        meta_tags = []
        for mid in merged_meta_ids:
            mid = str(mid).strip()
            if mid.startswith("P") and mid in person_map:
                meta_tags.append(brief_person(person_map[mid]))
            elif mid.startswith("D") and mid in doc_map:
                meta_tags.append(brief_doc(doc_map[mid]))

        # 저장되는 page_content(메타 요약 블록은 모델에 힌트로 제공)
        text = "\n".join([
            text_raw,
            f"메타: {', '.join([m for m in meta_tags if m])}"
        ]).strip()

        docs.append(Document(
            page_content=text,
            metadata={
                "title": title,
                "url": r.get("url",""),
                "meta_ids": merged_meta_ids,
                "meta_ids_provided": provided_meta_ids,
                "autolink_evidence": auto_evd[:5],  # 너무 길면 조금만 저장
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

    # 각 문서에 고유 인덱스 부여(검색 결과 매핑용)
    tmp_docs: List[Document] = []
    for i, d in enumerate(final_docs):
        md = dict(d.metadata); md["doc_id"] = i
        tmp_docs.append(Document(page_content=d.page_content, metadata=md))
    final_docs = tmp_docs

    emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vs = FAISS.from_documents(final_docs, emb)

    norm_contents = [ _norm(d.page_content) for d in final_docs ]
    norm_titles   = [ _norm(d.metadata.get("title","")) for d in final_docs ]
    norm_keywords = [ [_norm(k) for k in (d.metadata.get("keywords") or [])] for d in final_docs ]

    title_index = defaultdict(list)
    for i, d in enumerate(final_docs):
        title_index[d.metadata.get("title","")].append(i)  # 인덱스 저장

    glossary_rows = _safe_load_json(JSON_PATH_GLOSSARY)

    return final_docs, vs, norm_contents, norm_titles, norm_keywords, title_index, glossary_rows, doc_map, person_map, alias_records

# ======================
# 검색: 1순위(직접매칭) + 2순위(임베딩) + 엔티티 부스트
# ======================
def ids_from_query(query: str, alias_records: List[Dict[str,Any]], max_ids: int = 6) -> List[str]:
    """
    질의문에서 엔티티 ID 추출(가벼운 매칭). 긴 별칭 우선, 중복 제거.
    """
    q = unicodedata.normalize("NFKC", query or "")
    found, spans = [], []
    for rec in alias_records:
        if len(found) >= max_ids: break
        m = rec["re"].search(q)
        if not m:
            continue
        s, e = m.span()
        # 경계 대략 체크
        if s>0 and _is_word_char(q[s-1]):
            continue
        if e<len(q) and _is_word_char(q[e:e+1]):
            continue
        # 문헌 가드(질의는 보통 표식이 없으니 완화)
        found.append(rec["id"])
    # dedup
    uniq, seen = [], set()
    for x in found:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def retrieve(query: str,
             docs: List[Document],
             vs: FAISS,
             norm_contents: List[str],
             norm_titles: List[str],
             norm_keywords: List[List[str]],
             title_index,
             alias_records: List[Dict[str,Any]],
             k_total: int = 10,
             w_entity: float = 3.5) -> Tuple[List[int], List[int]]:
    """
    return: (primary_idxs, secondary_idxs)
    - primary: 직접 매칭 스코어 높은 순
    - secondary: 임베딩 상위에서 비중복 보충
    - 엔티티 부스트: 질의에서 정규화한 ID와 문서 meta_ids 교집합 크기 * w_entity
    """
    tokens = extract_terms(query)
    query_ids = set(ids_from_query(query, alias_records, max_ids=6))

    # 1) 직접 매칭 스코어
    scores = defaultdict(float)
    for i, (ct, tt, kws) in enumerate(zip(norm_contents, norm_titles, norm_keywords)):
        sc = 0.0
        for t in tokens:
            if not t: continue
            if t in ct:         sc += 6.0
            if t in tt:         sc += 7.0
            if any(t in kw for kw in kws): sc += 5.0
        # 엔티티 부스트
        if query_ids:
            mids = set(docs[i].metadata.get("meta_ids") or [])
            inter = len(query_ids & mids)
            if inter:
                sc += w_entity * inter
        if sc > 0:
            # 같은 제목 보강(제목 내 다른 파트 1개만)
            title = docs[i].metadata.get("title","")
            for j in title_index.get(title, [])[:2]:
                if j != i:
                    scores[j] += 2.0
            scores[i] += sc

    primary = [idx for idx,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    primary = primary[:min(k_total, 8)]  # 과다 방지

    # 2) 임베딩 보충 (doc_id로 안정 매핑)
    sec = []
    try:
        vec_docs = vs.as_retriever(search_kwargs={"k": k_total}).get_relevant_documents(query)
        for vd in vec_docs:
            did = vd.metadata.get("doc_id")
            if did is None:
                continue
            if did not in primary and did not in sec:
                sec.append(did)
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
    (docs, vs, norm_contents, norm_titles, norm_keywords, title_index,
     glossary_rows, doc_map, person_map, alias_records) = load_corpus_and_index()

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
                    q, docs, vs, norm_contents, norm_titles, norm_keywords, title_index,
                    alias_records, k_total=10, w_entity=3.5
                )

                # 컨텍스트 구성: 1순위 먼저, 부족하면 2순위 보충
                picked_idx = primary_idx + secondary_idx
                if not picked_idx:
                    st.error("관련 문서를 찾지 못했습니다.")
                    st.session_state.messages.append({"role":"assistant","content":"관련 문서를 찾지 못했습니다."})
                else:
                    context_parts = []
                    for i in picked_idx[:8]:
                        d = docs[i]
                        meta_lines = meta_briefs_for(d, doc_map, person_map, max_n=3)
                        meta_block = (" [관련 메타] " + " / ".join(meta_lines)) if meta_lines else ""
                        # 자동연결 증거를 보고 싶다면 아래 줄을 주석 해제
                        # debug_block = " [autolink] " + " || ".join(d.metadata.get("autolink_evidence", []))
                        context_parts.append(d.page_content + meta_block)

                    # 용어 보강은 1순위 본문 기준
                    primary_text = "\n".join([docs[i].page_content for i in primary_idx]) if primary_idx else ""
                    gloss = glossary_snips(glossary_rows, q, primary_text, max_n=3)
                    if gloss:
                        context_parts.append("[용어 보강] " + " | ".join(gloss))

                    # LLM 호출
                    ctx = "\n\n---\n\n".join(context_parts).strip()
                    prompt = ANSWER_PROMPT.format(context=ctx, question=q)
                    # ---- 직전 대화 히스토리(최근 N개 턴) 준비 ----
                    HISTORY_UTTER_MAX = 8  # 최근 8개 발화만 사용(원하면 조절)

                    lc_history = []
                    # st.session_state.messages에는 이번 user 질문까지 포함되어 있으므로 마지막 1개 제외
                    for m in st.session_state.messages[:-1][-HISTORY_UTTER_MAX:]:
                        if m["role"] == "user":
                            lc_history.append(HumanMessage(content=m["content"]))
                        else:
                            lc_history.append(AIMessage(content=m["content"]))

                    # ---- 시스템 메시지(간단 지시: 히스토리 반영 + RAG 컨텍스트 우선) ----
                    sys_msg = SystemMessage(
                        content="이전 대화 맥락을 반영해서 이어서 답하라. 다만 아래 제공되는 컨텍스트를 최우선 근거로 삼아라."
                    )

                    # ---- 프롬프트(기존 RAG 컨텍스트 포함된 prompt)를 '이번 사용자 메시지'로 넣기 ----
                    final_messages = [sys_msg] + lc_history + [HumanMessage(content=prompt)]

                    ans = llm.invoke(final_messages).content.strip()

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
