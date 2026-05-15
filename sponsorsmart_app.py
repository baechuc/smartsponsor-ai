import streamlit as st
import re, os, torch
import pandas as pd
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
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
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
st.markdown('<p class="subtitle">Sistem Pendukung Keputusan Penilaian Kelayakan Proposal Sponsorship</p>', unsafe_allow_html=True)
st.divider()

MODEL_PATH    = "./best_indobert_model"
SVM_PATH      = "./svm_model.pkl"
OCR_THRESHOLD = 50

RUBRIC_KEYWORDS = {
    "Exposure": [
        r"\b\d[\d.]*\s*(?:peserta|pengunjung|penonton|orang)\b",
        r"\b(?:instagram|youtube|tiktok|facebook|twitter)\b",
        r"\b(?:media sosial|live streaming|publikasi|promosi|liputan)\b",
        r"\b(?:followers|subscriber|views|reach)\b",
        r"\b(?:poster|banner|flyer|spanduk|baliho)\b",
    ],
    "Relevansi": [
        r"\b(?:teknologi|digital|startup|inovasi|bisnis)\b",
        r"\b(?:pendidikan|universitas|kampus|mahasiswa)\b",
        r"\b(?:olahraga|kesehatan|lifestyle|sport)\b",
        r"\b(?:seni|budaya|musik|festival|entertainment)\b",
        r"\b(?:sesuai|relevan|sejalan|mendukung)\s+(?:dengan|visi|misi|brand)\b",
    ],
    "Benefit": [
        r"\b(?:logo|branding|brand awareness|visibilitas)\b",
        r"\b(?:logo placement|official sponsor|title sponsor)\b",
        r"\b(?:booth|stand|pameran|aktivasi brand)\b",
        r"\b(?:mention|endorse|konten sponsor)\b",
        r"\b(?:tiket gratis|vip|akses eksklusif|goodie bag)\b",
    ],
    "Anggaran": [
        r"(?:rp|idr)\.?\s*[\d.,]{6,15}",
        r"(?:anggaran|biaya|dana|investasi|kontribusi)\s*[:=]?\s*(?:rp)?[\d.,]{6,15}",
        r"\b(?:paket)\s+(?:platinum|gold|silver|bronze)\b",
        r"\b(?:rab|rencana anggaran biaya)\b",
    ],
    "Kredibilitas": [
        r"\b(?:himpunan|bem|komunitas|lembaga|yayasan|pt\.|cv\.)\b",
        r"\b(?:susunan panitia|struktur organisasi|divisi)\b",
        r"\b(?:timeline|jadwal|rundown|susunan acara)\b",
        r"\b(?:contact person|cp|whatsapp|email)\b",
        r"\b(?:partner|mitra|bekerja sama|tahun lalu|edisi sebelumnya)\b",
    ]
}

THRESHOLDS = {
    "Exposure": 2, "Relevansi": 1, "Benefit": 2,
    "Anggaran": 1, "Kredibilitas": 3
}

@st.cache_resource
def load_bert_model():
    if not os.path.isdir(MODEL_PATH):
        return None, None, f"Folder '{MODEL_PATH}' tidak ditemukan."
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_svm_model():
    if not os.path.isfile(SVM_PATH):
        return None, f"File '{SVM_PATH}' tidak ditemukan."
    try:
        with open(SVM_PATH, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)

def extract_text(uploaded_file) -> tuple:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    full_text = ""
    method    = "standard"
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

def score_rubric(text: str) -> dict:
    text_lower = text.lower()
    scores, evidences = {}, {}

    for var, patterns in RUBRIC_KEYWORDS.items():
        matched = []
        for p in patterns:
            found = re.findall(p, text_lower, re.IGNORECASE)
            if found:
                matched.extend(found[:2])
        scores[var]    = 1 if len(set(matched)) >= THRESHOLDS[var] else 0
        evidences[var] = list(set(matched))[:3] if matched else []

    total = sum(scores.values())
    return {
        "scores": scores,
        "evidences": evidences,
        "total": total,
        "heuristic_label": "Layak" if total >= 3 else "Tidak Layak"
    }

def predict_bert(text: str, tokenizer, model) -> dict:
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

def render_score_chart(scores: dict):
    vars_  = list(scores.keys())
    vals   = list(scores.values())
    colors = ["#2ecc71" if v == 1 else "#e74c3c" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(vars_, vals, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1.3)
    ax.set_xlabel("Skor (0 = Tidak Terpenuhi, 1 = Terpenuhi)")
    ax.set_title("Skor Tiap Variabel Rubric")

    for bar, val in zip(bars, vals):
        label = "✓ Terpenuhi" if val == 1 else "✗ Tidak"
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10, fontweight="bold")

    green = mpatches.Patch(color="#2ecc71", label="Terpenuhi (1)")
    red   = mpatches.Patch(color="#e74c3c", label="Tidak Terpenuhi (0)")
    ax.legend(handles=[green, red], loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/target.png", width=80)
    st.title("⚙️ Pengaturan")
    model_choice = st.radio(
        "Pilih Model Prediksi:",
        ["IndoBERT (Rekomendasi)", "SVM + TF-IDF", "Keduanya (Bandingkan)"]
    )

    # ── Status ketersediaan model ──
    st.divider()
    st.markdown("### 🔌 Status Model")
    bert_available = os.path.isdir(MODEL_PATH)
    svm_available  = os.path.isfile(SVM_PATH)
    st.markdown(
        f"{'✅' if bert_available else '❌'} **IndoBERT** — "
        f"{'Siap' if bert_available else f'Tidak ditemukan (`{MODEL_PATH}`)'}"
    )
    st.markdown(
        f"{'✅' if svm_available else '❌'} **SVM** — "
        f"{'Siap' if svm_available else f'Tidak ditemukan (`{SVM_PATH}`)'}"
    )
    if not bert_available and not svm_available:
        st.warning("⚠️ Kedua model tidak ditemukan. Analisis akan menggunakan **heuristik rubric** saja.")
    elif not bert_available and "IndoBERT" in model_choice:
        st.warning("⚠️ IndoBERT tidak ditemukan. Pilih SVM atau jalankan notebook training terlebih dahulu.")
    elif not svm_available and "SVM" in model_choice:
        st.warning("⚠️ SVM tidak ditemukan. Pilih IndoBERT atau jalankan notebook training terlebih dahulu.")
    st.divider()
    st.markdown("### 📖 Tentang SponsorSmart AI")
    st.markdown("""
    Sistem ini membantu menilai kelayakan proposal sponsorship
    berdasarkan 5 variabel rubric:
    - 📡 **Exposure** — Jangkauan audiens
    - 🎯 **Relevansi** — Kesesuaian dengan sponsor
    - 🎁 **Benefit** — Keuntungan sponsor
    - 💰 **Anggaran** — Kejelasan biaya
    - 🏛️ **Kredibilitas** — Profesionalitas penyelenggara

    **Label:** Total Skor ≥ 3 → Layak
    """)

# ── Main Content ─────────────────────────────────────
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
        var_icons = {
            "Exposure": "📡", "Relevansi": "🎯", "Benefit": "🎁",
            "Anggaran": "💰", "Kredibilitas": "🏛️"
        }
        for i, (var, score) in enumerate(rubric_result["scores"].items()):
            with cols[i]:
                color  = "#d4edda" if score == 1 else "#f8d7da"
                status = "✓" if score == 1 else "✗"
                st.markdown(f"""
                <div style="background:{color}; padding:12px; border-radius:10px; text-align:center;">
                    <div style="font-size:1.5rem">{var_icons[var]}</div>
                    <div style="font-weight:bold; font-size:0.9rem">{var}</div>
                    <div style="font-size:1.8rem; font-weight:800">{status}</div>
                    <div style="font-size:0.8rem">Skor: {score}/1</div>
                </div>
                """, unsafe_allow_html=True)

        render_score_chart(rubric_result["scores"])
        st.markdown(f"**Total Skor Rubric: {rubric_result['total']}/5**")

        with st.expander("🔍 Detail Bukti per Variabel"):
            for var, evid in rubric_result["evidences"].items():
                icon = "✅" if rubric_result["scores"][var] == 1 else "❌"
                evid_str = str(evid) if evid else "Tidak ditemukan indikator"
                st.markdown(f"{icon} **{var}**: {evid_str}")

        st.divider()
        st.markdown("#### 🤖 Prediksi Model")
        bert_col, svm_col = st.columns(2)

        bert_result = {"label": None, "confidence": 0, "prob_layak": 0, "prob_tidak": 0}
        svm_label   = None

        if model_choice in ["IndoBERT (Rekomendasi)", "Keduanya (Bandingkan)"]:
            with bert_col:
                with st.spinner("🧠 Prediksi IndoBERT..."):
                    tokenizer, bert_model, bert_err = load_bert_model()
                    bert_result = predict_bert(full_text, tokenizer, bert_model)

                if bert_result["label"]:
                    badge_class = "layak-badge" if bert_result["label"] == "Layak" else "tidak-layak-badge"
                    st.markdown("**IndoBERT:**")
                    st.markdown(f'<span class="{badge_class}">{bert_result["label"]}</span>', unsafe_allow_html=True)
                    conf = bert_result["confidence"] * 100
                    st.progress(int(conf))
                    st.caption(f"Confidence: {conf:.1f}%")
                    st.caption(f"P(Layak)={bert_result['prob_layak']*100:.1f}% | P(Tidak Layak)={bert_result['prob_tidak']*100:.1f}%")
                else:
                    st.warning(f"⚠️ Model IndoBERT tidak dapat dimuat.\n\n**Penyebab:** {bert_err}\n\n💡 Jalankan notebook training untuk menghasilkan model, lalu letakkan folder `best_indobert_model/` di direktori yang sama dengan app ini.")
                    st.info("📋 Keputusan final akan menggunakan skor rubric heuristik.")

        if model_choice in ["SVM + TF-IDF", "Keduanya (Bandingkan)"]:
            with svm_col:
                with st.spinner("⚙️ Prediksi SVM..."):
                    svm_model, svm_err = load_svm_model()
                    if svm_model:
                        try:
                            svm_label = svm_model.predict([full_text])[0]
                            try:
                                svm_prob = svm_model.predict_proba([full_text])[0]
                                svm_conf = max(svm_prob) * 100
                            except AttributeError:
                                # Pipeline tidak support predict_proba
                                svm_conf = None
                            badge_class = "layak-badge" if svm_label == "Layak" else "tidak-layak-badge"
                            st.markdown("**SVM + TF-IDF:**")
                            st.markdown(f'<span class="{badge_class}">{svm_label}</span>', unsafe_allow_html=True)
                            if svm_conf is not None:
                                st.progress(int(svm_conf))
                                st.caption(f"Confidence: {svm_conf:.1f}%")
                            else:
                                st.caption("Confidence: tidak tersedia")
                        except Exception as e:
                            svm_label = None
                            st.warning(f"⚠️ Error saat prediksi SVM: {e}")
                    else:
                        st.warning(f"⚠️ Model SVM tidak dapat dimuat.\n\n**Penyebab:** {svm_err}\n\n💡 Jalankan notebook training untuk menghasilkan `svm_model.pkl`, lalu letakkan di direktori yang sama dengan app ini.")
                        st.info("📋 Keputusan final akan menggunakan skor rubric heuristik.")

        st.divider()

        # ── Keputusan Final ──────────────────────────────
        st.markdown("#### ⚖️ Keputusan Final")

        if model_choice == "IndoBERT (Rekomendasi)":
            final_label = bert_result["label"] if bert_result.get("label") else rubric_result["heuristic_label"]
        elif model_choice == "SVM + TF-IDF":
            final_label = svm_label if svm_label else rubric_result["heuristic_label"]
        else:
            if bert_result.get("label"):
                final_label = bert_result["label"]
            elif svm_label:
                final_label = svm_label
            else:
                final_label = rubric_result["heuristic_label"]

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
                saran = {
                    "Exposure"    : "Tambahkan data audiens (jumlah peserta, jangkauan media sosial, platform publikasi).",
                    "Relevansi"   : "Jelaskan keterkaitan acara dengan industri/brand sponsor secara eksplisit.",
                    "Benefit"     : "Cantumkan benefit konkret: logo placement, booth, mention medsos, goodie bag, dll.",
                    "Anggaran"    : "Sertakan nominal jelas (Rp) dan paket sponsorship (Gold/Silver/Bronze).",
                    "Kredibilitas": "Lengkapi identitas organisasi, susunan panitia, timeline, rundown, dan kontak PIC.",
                }
                for var in tidak_terpenuhi:
                    st.markdown(f"**{var}:** {saran.get(var, '')}")

else:
    st.info("👆 Upload file PDF proposal sponsorship untuk memulai analisis.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>📡</h2><b>Exposure</b>
            <p>Jangkauan audiens & media</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🎁</h2><b>Benefit</b>
            <p>Keuntungan nyata untuk sponsor</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🏛️</h2><b>Kredibilitas</b>
            <p>Profesionalitas penyelenggara</p>
        </div>""", unsafe_allow_html=True)
