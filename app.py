# -*- coding: utf-8 -*-
"""
한신 초등 은유 이야기 기계 — Hugging Face Inference API 버전
- OpenAI 종속성 제거, requests 기반 HF 호출
- 학생 정보(반/번호/이름) + 로고
- "주인공 만들기" → 주인공 설명(초3 문체, 한국어)
- 8칸 이야기(2·4·6 자동 이어쓰기: 지금까지 내용 전부 고려, 200~300자)
- 8칸 모두 채우면 하나의 완성 이야기로 정리
필수: Streamlit Secrets에 아래 추가
HUGGINGFACEHUB_API_TOKEN = "hf_xxx..."
(선택) HF_MODEL = "EleutherAI/polyglot-ko-1.3b"  # 기본값
"""
import re
import os
import requests
import streamlit as st

# -----------------------------
# 기본 설정 + 로고
# -----------------------------
st.set_page_config(page_title="한신 초등 은유 이야기 기계 (HF)", page_icon="✨")
if os.path.exists("logo.PNG"):
    st.image("logo.PNG", width=120)
st.title("✨ 한신 초등학교 친구들의 이야기 실력을 볼까요?")
st.caption("좋아하는 단어로 주인공을 먼저 만들고, 그 다음에 이야기를 이어가요! (Hugging Face API)")


# -----------------------------
# Hugging Face Inference API 설정
# -----------------------------
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL = st.secrets.get("HF_MODEL", "EleutherAI/polyglot-ko-1.3b")

def hf_generate(prompt: str, max_new_tokens: int = 220, temperature: float = 0.7, top_p: float = 0.9):
    """
    Hugging Face Inference API 호출 (text-generation)
    """
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN이 설정되지 않았습니다. Streamlit Secrets에 추가해 주세요.")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 503:
        # 모델이 아직 로딩 중인 경우
        raise RuntimeError("모델이 준비 중이에요. 잠시 뒤 다시 시도해 주세요. (HF 503)")
    if resp.status_code == 429:
        raise RuntimeError("Hugging Face 호출이 너무 잦습니다(429). 잠시 뒤 다시 시도해 주세요.")
    if not resp.ok:
        raise RuntimeError(f"Hugging Face API 오류: {resp.status_code} - {resp.text}")

    data = resp.json()
    # 응답 형식 방어적 처리
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    # text-generation 계열 외 모델 대응
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    return str(data)


# -----------------------------
# 학생 정보 입력
# -----------------------------
st.subheader("👧 학생 정보 입력")
col1, col2, col3 = st.columns(3)
cls = col1.text_input("학급 (예: 3-2)")
num = col2.text_input("번호")
name = col3.text_input("이름")


# -----------------------------
# 금칙어 목록 (좋아하는 단어 필터)
# -----------------------------
BANNED_PATTERNS = [
    r"살인", r"죽이", r"폭력", r"피바다", r"학대", r"총", r"칼", r"폭탄",
    r"kill", r"murder", r"gun", r"knife", r"blood", r"assault", r"bomb",
    r"성\s*행위", r"야동", r"포르노", r"음란", r"가슴", r"성기", r"자위",
    r"porn", r"sex", r"xxx", r"nude", r"naked",
]
BAN_RE = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)

def words_valid(words):
    for w in words:
        if not w:
            return False, "단어 3개를 모두 입력해 주세요."
        if BAN_RE.search(w):
            return False, "적절하지 않은 단어입니다. 다시 입력해 주세요."
    return True, "OK"


# -----------------------------
# 1️⃣ 주인공 만들기 (스토리 입력 전에 먼저 생성/표시)
# -----------------------------
st.subheader("1️⃣ 좋아하는 단어 3개로 주인공 만들기")
c1, c2, c3 = st.columns(3)
w1 = c1.text_input("단어 1", max_chars=12)
w2 = c2.text_input("단어 2", max_chars=12)
w3 = c3.text_input("단어 3", max_chars=12)

st.session_state.setdefault("character_desc", "")

if st.button("주인공 만들기 👤✨"):
    words = [w1.strip(), w2.strip(), w3.strip()]
    ok, msg = words_valid(words)
    if not ok:
        st.error(msg)
    else:
        prompt = (
            "아래 세 단어를 모두 사용해서 초등학교 3학년이 읽기 쉬운 문체로, "
            "주인공의 이름, 성격, 좋아하는 일, 사는 곳을 3~4문장으로 소개해 주세요.\n"
            f"단어: {words[0]}, {words[1]}, {words[2]}\n"
            "말투 예시: '이름은 ○○예요. 밝고 친절한 성격이에요. ...' 처럼.\n"
            "부드럽고 따뜻한 문장으로 써 주세요. 한국어로 작성해 주세요."
        )
        try:
            desc = hf_generate(prompt, max_new_tokens=220, temperature=0.8, top_p=0.9)
            # 출력 정리 (불필요한 따옴표/마크다운 제거 정도)
            desc = desc.replace("###", "").strip()
            st.session_state["character_desc"] = desc
            st.success("💫 주인공이 완성되었어요!")
        except Exception as e:
            st.error(f"주인공 생성 중 문제가 발생했어요: {e}")

# 주인공 정보 표시
if st.session_state["character_desc"]:
    st.markdown("### 👤 주인공 소개")
    st.write(st.session_state["character_desc"])
else:
    st.info("먼저 단어 3개로 주인공을 만들어 주세요. 그 다음에 이야기 칸이 열려요!")

# -----------------------------
# 2️⃣ 8단 이야기 — 주인공이 생긴 뒤에만 노출
# -----------------------------
if st.session_state["character_desc"]:
    st.divider()
    st.subheader("2️⃣ 주인공의 이야기를 써 볼까요? ✍️")

    TITLES = [
        "옛날에", "그리고 매일", "그러던 어느 날",
        "그래서", "그래서", "그래서",
        "마침내", "그날 이후",
    ]

    for i in range(8):
        st.session_state.setdefault(f"story_{i}", "")
        st.session_state.setdefault(f"auto_{i}", False)

    for i, title in enumerate(TITLES):
        st.markdown(f"#### {title}")
        if i in [0, 2, 4, 6, 7]:
            st.session_state[f"story_{i}"] = st.text_area(
                f"{title} 내용을 적어보세요",
                value=st.session_state[f"story_{i}"],
                height=90,
                key=f"story_input_{i}",
            )
        else:
            if st.button(f"{title} 자동 이어쓰기 🪄", key=f"auto_btn_{i}"):
                prev_all_list = []
                for j in range(i):
                    key_story = f"story_{j}"
                    if st.session_state[key_story]:
                        prev_all_list.append(st.session_state[key_story])
                prev_all = " ".join(prev_all_list).strip()

                if not prev_all:
                    st.warning("이전까지의 이야기를 먼저 적어 주세요!")
                else:
                    character = st.session_state.get("character_desc", "")
                    prompt = (
                        "아래의 지금까지의 이야기를 바탕으로, "
                        f"'{title}'에 어울리는 다음 장면을 200~300자 한국어로 이어서 써 주세요.\n"
                        "초등학교 3학년이 이해하기 쉬운, 부드럽고 따뜻한 문체로 작성해 주세요.\n\n"
                        f"지금까지의 이야기:\n\"\"\"{prev_all}\"\"\"\n\n"
                        f"주인공 정보:\n{character}\n"
                    )
                    try:
                        auto_text = hf_generate(prompt, max_new_tokens=260, temperature=0.85, top_p=0.9)
                        st.session_state[f"story_{i}"] = auto_text.strip()
                        st.session_state[f"auto_{i}"] = True
                        st.info("자동으로 이어썼어요 ✨")
                    except Exception as e:
                        st.error(f"이어쓰기 중 문제가 발생했어요: {e}")

            st.text_area(
                f"{title} 자동 생성된 내용",
                value=st.session_state[f"story_{i}"],
                height=120,
                disabled=True,
                key=f"auto_output_{i}",
            )

    # -----------------------------
    # 3️⃣ 8칸 모두 작성 시: 하나의 완성 이야기로 합치기
    # -----------------------------
    if all(st.session_state[f"story_{i}"].strip() for i in range(8)):
        st.divider()
        st.subheader("🎉 완성된 이야기")

        parts = []
        for i in range(8):
            key_story = f"story_{i}"
            parts.append(f"**{TITLES[i]}**\n{st.session_state[key_story]}")
        story_text = "\n\n".join(parts)

        # 간단한 마무리 다듬기(요약 없이 합치기) — HF로 리라이트하고 싶으면 주석 해제
        try:
            polish_prompt = (
                "다음 8단 이야기를 자연스럽게 한 편의 이야기로 다듬어 주세요. "
                "초등학교 3학년 수준의 쉬운 한국어 문장으로 바꿔 주세요. 너무 길게 늘리지 마세요.\n\n"
                f"{story_text}"
            )
            final_story = hf_generate(polish_prompt, max_new_tokens=300, temperature=0.6, top_p=0.9)
        except Exception:
            final_story = story_text

        st.write(final_story)
        safe_name = f"{cls}_{num}_{name}_story.txt".replace(" ", "_")
        st.download_button(
            "📥 완성된 이야기 저장하기 (txt)",
            data=final_story,
            file_name=safe_name if safe_name != "__story.txt" else "my_story.txt",
            mime="text/plain",
        )
