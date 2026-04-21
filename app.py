import streamlit as st
import pdfplumber

from core.pdf_chat import build_index, search_pdf
from core.nlp_pipeline import process_query
from core.retriever import search_indian_kanoon
from core.legal_reasoner import generate_output

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Legal Intelligence Engine", layout="wide")

# -------- CUSTOM CSS --------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}

.title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    color: gray;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

button {
    background-color: #0d6efd;
    color: white;
    border-radius: 8px;
    height: 40px;
}

</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown('<div class="title">⚖️ Legal Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system for analyzing Indian legal documents & queries</div>', unsafe_allow_html=True)

# -------- SESSION STATE --------
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "auto_output" not in st.session_state:
    st.session_state.auto_output = ""

# -------- SAMPLE BUTTONS --------
st.subheader("💡 Try Sample Queries")

col1, col2, col3 = st.columns(3)

if col1.button("Salary not paid"):
    st.session_state.query_text = "My employer is not paying my salary"

if col2.button("Property dispute"):
    st.session_state.query_text = "There is a property dispute"

if col3.button("Criminal case"):
    st.session_state.query_text = "Someone filed a false FIR against me"

# -------- INPUT TYPE --------
option = st.radio("Choose Input Type", ["Text Query", "Upload PDF"])

# -------- TEXT INPUT --------
query = ""
if option == "Text Query":
    query = st.text_area("Enter your legal issue", value=st.session_state.query_text)

    if st.button("Analyze ⚖️") and query:
        with st.spinner("Analyzing..."):

            domain = process_query(query)
            context = ""
            cases = search_indian_kanoon(query)

            result = generate_output(query, context, cases[:3], domain)

        st.session_state.auto_output = result

# -------- PDF INPUT --------
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload Legal PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

            build_index(text)

        st.session_state.pdf_ready = True
        st.success("PDF ready for analysis ✅")

        # AUTO ANALYSIS
        default_query = "Analyze this legal document"

        domain = process_query(default_query)
        context = search_pdf(default_query)
        cases = search_indian_kanoon(default_query)

        result = generate_output(default_query, context, cases[:3], domain)

        st.session_state.auto_output = result

# -------- OUTPUT SECTION --------
def format_output(text):
    sections = {
        "Legal Issue": "",
        "Jurisdiction": "",
        "Key Facts": "",
        "Primary Precedents": "",
        "Live Court Updates": "",
        "Legal Analysis (IRAC)": ""
    }

    current = None

    for line in text.split("\n"):
        line = line.strip()

        for key in sections.keys():
            if key.lower() in line.lower():
                current = key
                continue

        if current:
            sections[current] += line + "\n"

    return sections

# -------- DISPLAY OUTPUT --------
if st.session_state.auto_output:

    st.markdown("---")
    st.subheader("📊 Legal Intelligence Output")

    sections = format_output(st.session_state.auto_output)

    for title, content in sections.items():
        st.markdown(f"""
        <div class="card">
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
        """, unsafe_allow_html=True)

# -------- CHAT WITH PDF --------
if option == "Upload PDF" and st.session_state.pdf_ready:

    st.markdown("---")
    st.subheader("💬 Ask about this document")

    col1, col2 = st.columns([8, 1])

    with col1:
        user_query = st.text_input("Type your question...")

    with col2:
        send = st.button("➤")

    if send and user_query:
        with st.spinner("Analyzing..."):

            domain = process_query(user_query)
            context = search_pdf(user_query)
            cases = search_indian_kanoon(user_query)

            result = generate_output(user_query, context, cases[:3], domain)

        st.markdown("### 📊 Legal Response")
        st.markdown(result)