import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page Configuration
st.set_page_config(
    page_title="Advanced ATS Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Charcoal Dark Theme & Blue Button Styling matching the reference
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a1a;
        color: #e5e5e5;
    }
    .app-header {
        background: #242424;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .panel-card {
        background: #242424;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 20px;
    }
    h1, h2, h3, h4, p, label {
        color: #f5f5f5 !important;
    }
    /* Style primary action button to match blue theme */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        height: 48px !important;
        width: 100% !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    }
</style>
""", unsafe_allow_html=True)

SKILL_TAXONOMY = {
    "Languages": {"python", "java", "javascript", "typescript", "c", "cpp", "c++", "go", "rust", "sql"},
    "Frameworks & Web": {"react", "reactjs", "node", "nodejs", "express", "fastapi", "flask", "django", "tailwind", "nextjs"},
    "Databases": {"mongodb", "mysql", "postgresql", "postgres", "sqlite", "redis", "firebase"},
    "Tools & DevOps": {"git", "github", "docker", "kubernetes", "aws", "gcp", "azure", "postman", "linux", "ci/cd"}
}

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_categorized_skills(text):
    text_lower = text.lower()
    found_skills = {cat: set() for cat in SKILL_TAXONOMY}
    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills[category].add(skill)
    return found_skills

# Header
st.markdown("""
<div class="app-header">
    <h1>Advanced ATS Analyzer</h1>
    <p>Semantic Matching • Skill Taxonomy • AI Suggestions</p>
</div>
""", unsafe_allow_html=True)

# Dual-Panel Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Input Configuration")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=[".pdf"])
    jd_input = st.text_area("Job Specification", height=180, placeholder="Paste target job description skills and requirements here...")
    analyze_btn = st.button("Execute Match Matrix", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Intelligence Telemetry")
    
    if analyze_btn:
        if uploaded_file is None or not jd_input.strip():
            st.warning("Please upload a PDF resume and provide a job description.")
        else:
            with st.spinner("Running semantic analysis and taxonomy matching..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.error("Could not extract text from the uploaded PDF.")
                else:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_input])
                    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    semantic_score = round(float(similarity_score) * 100, 2)
                    
                    resume_skills = extract_categorized_skills(resume_text)
                    jd_skills = extract_categorized_skills(jd_input)
                    
                    st.markdown(f"### Semantic Match Score: `{semantic_score}%`")
                    st.markdown("---")
                    
                    all_missing = []
                    for category in SKILL_TAXONOMY:
                        req = jd_skills[category]
                        has = resume_skills[category]
                        matched = req.intersection(has)
                        missing = req - has
                        all_missing.extend(missing)
                        
                        st.markdown(f"**{category}**")
                        st.markdown(f"- Matched: `{', '.join(list(matched)) if matched else 'None'}`")
                        st.markdown(f"- Missing: `{', '.join(list(missing)) if missing else 'None'}`")
                    
                    st.markdown("---")
                    st.markdown("#### Actionable Recommendations")
                    if all_missing:
                        st.markdown(f"- **Keyword Optimization:** Integrate missing skills like `{', '.join(list(all_missing)[:5])}` into your project descriptions.")
                    else:
                        st.markdown("- Great Job! Your resume matches all core technical requirements.")
                    
                    with st.expander("Extracted Resume Stream"):
                        st.text(resume_text[:1200] + "...")
    else:
        st.info("Submit inputs on the left panel to execute telemetry analysis.")
    st.markdown('</div>', unsafe_allow_html=True)