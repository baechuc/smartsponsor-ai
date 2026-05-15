import streamlit as st
import re, os, torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

st.set_page_config(
    page_title="SponsorSmart AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.5rem;
    }
    .subtitle { text-align: center; color: #666; margin-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #667eea; margin-bottom: 0.5rem;
    }
    .layak-badge {
        background: #d4edda; color: #155724;
        padding: 0.8rem 2rem; border-radius: 25px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin: 1rem 0;
    }
    .tidak-layak-badge {
        background: #f8d7da; color: #721c24;
        padding: 0.8rem 2rem; border-radius: 25px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎯 SponsorSmart AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Pendukung Keputusan Penilaian Kelayakan Proposal Sponsorship</p>',
            unsafe_allow_html=True)
st.divider()

MODEL_PATH    = "./best_indobert_model"
SVM_PATH      = "./svm_model.pkl"
OCR_THRESHOLD = 50

RUBRIC_KEYWORDS = {
    "Exposure": [
        r"\b\d[\d.]*\s*(?:peserta|pengunjung|penonton|orang|hadirin)\b",
        r"\b(?:instagram|youtube|tiktok|facebook|twitter|linkedin)\b",
        r"\b(?:media sosial|live streaming|publikasi|promosi|liputan)\b",
        r"\b(?:followers|subscriber|views|reach|tayangan)\b",
        r"\b(?:poster|banner|flyer|spanduk|baliho|pamflet)\b",
    ],
    "Relevansi": [
        r"\b(?:teknologi|digital|startup|inovasi|bisnis|korporat)\b",
        r"\b(?:pendidikan|universitas|kampus|mahasiswa|akademis)\b",
        r"\b(?:olahraga|kesehatan|lifestyle|sport|atletik)\b",
        r"\b(?:seni|budaya|musik|festival|entertainment|hiburan)\b",
        r"\b(?:sesuai|relevan|sejalan|mendukung)\s+(?:dengan|visi|misi|brand)\b",
    ],
    "Benefit": [
        r"\b(?:logo|branding|brand awareness|visibilitas)\b",
        r"\b(?:logo placement|official sponsor|title sponsor|presenting sponsor)\b",
        r"\b(?:booth|stand|pameran|aktivasi brand|exhibition)\b",
        r"\b(?:mention|endorse|konten sponsor|publikasi sponsor)\b",
        r"\b(?:tiket gratis|vip|akses eksklusif|goodie bag|merchandise)\b",
    ],
    "Anggaran": [
        r"(?:rp|idr)\.?\s*[\d.,]{6,15}",
        r"(?:anggaran|biaya|dana|investasi|kontribusi|nominal)\s*[:=]?\s*(?:rp)?[\d.,]{6,15}",
        r"\b(?:paket)\s+(?:platinum|gold|silver|bronze|diamond)\b",
        r"\b(?:rab|rencana anggaran biaya|kebutuhan dana)\b",
    ],
    "Kredibilitas": [
        r"\b(?:himpunan|bem|osis|komunitas|lembaga|yayasan|pt\.|cv\.|organisasi)\b",
        r"\b(?:susunan panitia|struktur organisasi|divisi|sie)\b",
        r"\b(?:ketua|sekretaris|bendahara|koordinator|penanggung jawab)\b",
        r"\b(?:timeline|jadwal|rundown|susunan acara|agenda)\b",
        r"\b(?:contact person|cp|whatsapp|narahubung|email|telp)\b",
    ]
}

THRESHOLDS = {
    "Exposure": 2, "Relevansi": 1, "Benefit": 2,
    "Anggaran": 1, "Kredibilitas": 3
}

SARAN = {
    "Exposure"    : "Tambahkan data audiens (jumlah peserta, jangkauan media sosial, platform publikasi).",
    "Relevansi"   : "Jelaskan keterkaitan acara dengan industri/brand sponsor secara eksplisit.",
    "Benefit"     : "Cantumkan benefit konkret: logo placement, booth, mention medsos, goodie bag, dll.",
    "Anggaran"    : "Sertakan nominal jelas (Rp) dan paket sponsorship (Gold/Silver/Bronze).",
    "Kredibilitas": "Lengkapi identitas organisasi, susunan panitia, timeline, rundown, dan kontak PIC.",
}

VAR_ICONS = {
    "Exposure":"📡", "Relevansi":"🎯", "Benefit":"🎁",
    "Anggaran":"💰", "Kredibilitas":"🏛️"
}

@st.cache_resource
def load_bert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except:
        return None, None

@st.cache_resource
def load_svm_model():
    try:
        with open(SVM_PATH, "rb") as f:
            return pickle.load(f)
    except:
        return None

def extract_text(uploaded_file):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    full_text = ""
    method    = "pdfplumber"
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages[:10]:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except:
        pass
    if len(full_text.strip()) < OCR_THRESHOLD:
        method = "ocr"
        try:
            images = convert_from_path(tmp_path, first_page=1, last_page=3)
            for img in images:
                full_text += pytesseract.image_to_string(img, lang="ind") + "\n"
        except:
            full_text = ""
    os.unlink(tmp_path)
    return full_text.strip(), method

def score_rubric(text):
    text_lower = text.lower()
    scores, evidences = {}, {}
    for var, patterns in RUBRIC_KEYWORDS.items():
        matched = []
        for p in patterns:
            found = re.findall(p, text_lower, re.IGNORECASE)
            if found:
                matched.extend(found[:2])
        unique_matched = list(set(matched))
        scores[var]    = 1 if len(unique_matched) >= THRESHOLDS[var] else 0
        evidences[var] = unique_matched[:3] if unique_matched else []
    total = sum(scores.values())
    return {
        "scores"         : scores,
        "evidences"      : evidences,
        "total"          : total,
        "heuristic_label": "Layak" if total >= 3 else "Tidak Layak"
    }

def predict_bert(text, tokenizer, model):
    if tokenizer is None or model is None:
        return {"label": None, "confidence": None, "prob_layak": 0, "prob_tidak": 0}
    inputs = tokenizer(
        text, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    label_idx  = int(torch.argmax(outputs.logits))
    labels_map = {0: "Tidak Layak", 1: "Layak"}
    return {
        "label"     : labels_map[label_idx],
        "confidence": max(probs),
        "prob_layak": probs[1],
        "prob_tidak": probs[0]
    }

def render_score_chart(scores):
    vars_  = list(scores.keys())
    vals   = list(scores.values())
    colors = ["#2ecc71" if v == 1 else "#e74c3c" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(vars_, vals, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Skor (0 = Tidak Terpenuhi, 1 = Terpenuhi)")
    ax.set_title("Skor Tiap Variabel Rubric")
    for bar, val in zip(bars, vals):
        lbl = "✓ Terpenuhi" if val == 1 else "✗ Tidak"
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=10, fontweight="bold")
    green = mpatches.Patch(color="#2ecc71", label="Terpenuhi (1)")
    red   = mpatches.Patch(color="#e74c3c", label="Tidak Terpenuhi (0)")
    ax.legend(handles=[green, red], loc="lower right")
    plt.tight_layout()
    return fig

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/target.png", width=80)
    st.title("⚙️ Pengaturan")
    model_choice = st.radio(
        "Pilih Model Prediksi:",
        ["IndoBERT (Rekomendasi)", "SVM + TF-IDF", "Keduanya (Bandingkan)"]
    )
    st.divider()
    st.markdown("### 📖 Tentang SponsorSmart AI")
    st.markdown("""
    Sistem ini membantu menilai kelayakan proposal sponsorship
    berdasarkan **5 variabel rubric**:
    - 📡 **Exposure** — Jangkauan audiens
    - 🎯 **Relevansi** — Kesesuaian dengan sponsor
    - 🎁 **Benefit** — Keuntungan sponsor
    - 💰 **Anggaran** — Kejelasan biaya
    - 🏛️ **Kredibilitas** — Profesionalitas penyelenggara

    **Label:** Total Skor ≥ 3 → **Layak**
    """)

# ── Main ─────────────────────────────────────────────
st.subheader("📤 Upload Proposal Sponsorship")
uploaded_file = st.file_uploader(
    "Upload file PDF proposal sponsorship",
    type=["pdf"],
    help="Format PDF, maksimal 50MB"
)

if uploaded_file is not None:
    st.success(f"✅ File berhasil diupload: **{uploaded_file.name}**")

    with st.spinner("🔍 Mengekstrak teks dari PDF..."):
        full_text, method = extract_text(uploaded_file)

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("👁️ Preview Teks Hasil Ekstraksi", expanded=False):
            if full_text:
                preview = full_text[:2000] + "..." if len(full_text) > 2000 else full_text
                st.text_area("Teks Proposal:", preview, height=250)
            else:
                st.error("Teks tidak dapat diekstrak dari file ini.")
    with col2:
        st.metric("Metode Ekstraksi", method.upper())
        st.metric("Jumlah Kata",      len(full_text.split()))
        st.metric("Jumlah Karakter",  len(full_text))

    if not full_text or len(full_text.split()) < 30:
        st.error("⚠️ Teks proposal terlalu sedikit untuk dianalisis.")
        st.stop()

    st.divider()

    if st.button("🚀 Analisis Proposal", type="primary", use_container_width=True):
        st.divider()
        st.subheader("📊 Hasil Analisis")

        with st.spinner("📋 Menghitung skor rubric..."):
            rubric_result = score_rubric(full_text)

        st.markdown("#### 📋 Skor Tiap Variabel Rubric")
        cols = st.columns(5)
        for i, (var, score) in enumerate(rubric_result["scores"].items()):
            with cols[i]:
                color  = "#d4edda" if score == 1 else "#f8d7da"
                status = "✓" if score == 1 else "✗"
                st.markdown(f"""
                <div style="background:{color};padding:12px;border-radius:10px;text-align:center;">
                    <div style="font-size:1.5rem">{VAR_ICONS[var]}</div>
                    <div style="font-weight:bold;font-size:0.9rem">{var}</div>
                    <div style="font-size:1.8rem;font-weight:800">{status}</div>
                    <div style="font-size:0.8rem">Skor: {score}/1</div>
                </div>
                """, unsafe_allow_html=True)

        st.pyplot(render_score_chart(rubric_result["scores"]))
        st.markdown(f"**Total Skor Rubric: {rubric_result['total']}/5**")

        with st.expander("🔍 Detail Bukti per Variabel"):
            for var, evid in rubric_result["evidences"].items():
                icon = "✅" if rubric_result["scores"][var] == 1 else "❌"
                st.markdown(f"{icon} **{var}**: {evid if evid else 'Tidak ditemukan indikator'}")

        st.divider()

        # ── Prediksi Model ────────────────────────────
        st.markdown("#### 🤖 Prediksi Model")

        # ✅ FIX: Inisialisasi variabel SEBELUM blok kondisi
        bert_result = {"label": None, "confidence": None, "prob_layak": 0, "prob_tidak": 0}
        svm_label   = None
        svm_conf    = 0

        bert_col, svm_col = st.columns(2)

        if model_choice in ["IndoBERT (Rekomendasi)", "Keduanya (Bandingkan)"]:
            with bert_col:
                with st.spinner("🧠 Prediksi IndoBERT..."):
                    tokenizer, bert_model = load_bert_model()
                    bert_result = predict_bert(full_text, tokenizer, bert_model)
                if bert_result["label"]:
                    badge = "layak-badge" if bert_result["label"] == "Layak" else "tidak-layak-badge"
                    st.markdown("**IndoBERT:**")
                    st.markdown(f'<span class="{badge}">{bert_result["label"]}</span>',
                                unsafe_allow_html=True)
                    conf = int(bert_result["confidence"] * 100)
                    st.progress(conf)
                    st.caption(f"Confidence: {conf}%")
                    st.caption(
                        f"P(Layak)={bert_result[chr(39)+'prob_layak'+chr(39)]*100:.1f}% | "
                        f"P(Tidak Layak)={bert_result[chr(39)+'prob_tidak'+chr(39)]*100:.1f}%"
                    )
                else:
                    st.warning("Model IndoBERT tidak ditemukan. Pakai hasil rubric saja.")

        if model_choice in ["SVM + TF-IDF", "Keduanya (Bandingkan)"]:
            with svm_col:
                with st.spinner("⚙️ Prediksi SVM..."):
                    svm_model = load_svm_model()
                    if svm_model:
                        svm_label = svm_model.predict([full_text])[0]
                        svm_prob  = svm_model.predict_proba([full_text])[0]
                        svm_conf  = int(max(svm_prob) * 100)
                        badge = "layak-badge" if svm_label == "Layak" else "tidak-layak-badge"
                        st.markdown("**SVM + TF-IDF:**")
                        st.markdown(f'<span class="{badge}">{svm_label}</span>',
                                    unsafe_allow_html=True)
                        st.progress(svm_conf)
                        st.caption(f"Confidence: {svm_conf}%")
                    else:
                        st.warning("Model SVM tidak ditemukan.")

        st.divider()

        # ── Keputusan Final ───────────────────────────
        # ✅ FIX: Semua variabel sudah pasti terdefinisi
        st.markdown("#### ⚖️ Keputusan Final")

        if model_choice == "IndoBERT (Rekomendasi)":
            final_label = bert_result["label"] or rubric_result["heuristic_label"]
        elif model_choice == "SVM + TF-IDF":
            final_label = svm_label or rubric_result["heuristic_label"]
        else:
            final_label = (bert_result["label"] or svm_label or
                           rubric_result["heuristic_label"])

        badge_class = "layak-badge" if final_label == "Layak" else "tidak-layak-badge"
        label_text  = "✅ LAYAK" if final_label == "Layak" else "❌ TIDAK LAYAK"

        col_dec, col_reason = st.columns([1, 2])
        with col_dec:
            st.markdown(
                f'<div style="text-align:center"><span class="{badge_class}">{label_text}</span></div>',
                unsafe_allow_html=True
            )
        with col_reason:
            st.markdown("**Alasan Keputusan:**")
            terpenuhi       = [v for v, s in rubric_result["scores"].items() if s == 1]
            tidak_terpenuhi = [v for v, s in rubric_result["scores"].items() if s == 0]
            if terpenuhi:
                st.success(f"✅ Variabel terpenuhi: {', '.join(terpenuhi)}")
            if tidak_terpenuhi:
                st.error(f"❌ Variabel tidak terpenuhi: {', '.join(tidak_terpenuhi)}")
            if final_label == "Layak":
                st.info("💡 Proposal memenuhi minimal 3 dari 5 kriteria kelayakan sponsorship.")
            else:
                st.warning("💡 Proposal perlu diperkuat — kurang dari 3 kriteria terpenuhi.")

        if tidak_terpenuhi:
            with st.expander("💡 Saran Perbaikan Proposal"):
                for var in tidak_terpenuhi:
                    st.markdown(f"**{var}:** {SARAN.get(var, \'\')}")

else:
    st.info("👆 Upload file PDF proposal sponsorship untuk memulai analisis.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <h2>📡</h2><b>Exposure</b><p>Jangkauan audiens & media</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <h2>🎁</h2><b>Benefit</b><p>Keuntungan nyata untuk sponsor</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <h2>🏛️</h2><b>Kredibilitas</b><p>Profesionalitas penyelenggara</p>
        </div>""", unsafe_allow_html=True)
