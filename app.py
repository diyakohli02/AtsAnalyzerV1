import gradio as gr
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import re

print("Loading AI Semantic Model & Skill Taxonomy Engine...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Structured Skill Taxonomy Database
SKILL_TAXONOMY = {
    "Languages": {"python", "java", "javascript", "typescript", "c", "cpp", "c++", "go", "rust", "sql"},
    "Frameworks & Web": {"react", "reactjs", "node", "nodejs", "express", "fastapi", "flask", "django", "tailwind", "nextjs"},
    "Databases": {"mongodb", "mysql", "postgresql", "postgres", "sqlite", "redis", "firebase"},
    "Tools & DevOps": {"git", "github", "docker", "kubernetes", "aws", "gcp", "azure", "postman", "linux", "ci/cd"}
}

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    text = ""
    with pdfplumber.open(pdf_file.name) as pdf:
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

def analyze_resume_comprehensive(pdf_file, job_description):
    if pdf_file is None or not job_description.strip():
        return "Please upload a PDF resume and provide a job description.", ""
    
    resume_text = extract_text_from_pdf(pdf_file)
    if not resume_text.strip():
        return "Could not extract text from the PDF.", ""
    
    # 1. Semantic Embedding Match Score
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    semantic_score = round(util.cos_sim(resume_embedding, jd_embedding).item() * 100, 2)
    
    # 2. Taxonomy Skill Gap Analysis
    resume_skills = extract_categorized_skills(resume_text)
    jd_skills = extract_categorized_skills(job_description)
    
    report = f"## Comprehensive ATS & Skill Gap Report\n\n"
    report += f"### **Semantic Similarity Score:** `{semantic_score}%`\n\n---\n\n"
    
    all_missing = []
    for category in SKILL_TAXONOMY:
        req = jd_skills[category]
        has = resume_skills[category]
        matched = req.intersection(has)
        missing = req - has
        all_missing.extend(missing)
        
        report += f"#### **{category}**\n"
        report += f"- Matched: `{', '.join(list(matched)) if matched else 'None'}`\n"
        report += f"- Missing: `{', '.join(list(missing)) if missing else 'None'}`\n\n"
        
    # 3. Actionable Improvement Suggestions
    report += "---\n\n### Actionable Recommendations:\n\n"
    if all_missing:
        report += f"- **Keyword Optimization:** Integrate missing skills like `{', '.join(list(all_missing)[:5])}` into your project descriptions or technical skills section.\n"
        report += f"- **Context Alignment:** Ensure your resume bullet points explicitly reflect core technologies.\n"
    else:
        report += "- Great Job! Your resume matches all core technical requirements from the taxonomy.\n"
        
    return report, resume_text[:1200] + "..."

# Dark Charcoal Theme with Modern Dual-Panel Layout Styling
custom_css = """
body, .gradio-container {
    background-color: #1a1a1a !important;
    color: #e5e5e5 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Header Container */
.app-header {
    background: #242424 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
}
.app-header h1 {
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}
.app-header p {
    color: #a3a3a3 !important;
    font-size: 14px !important;
}

/* Left & Right Panel Cards */
.panel-card {
    background: #242424 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Form Controls & Textareas Dark Charcoal Override */
textarea, input, .file-drop, div[data-testid="file-upload"] {
    background-color: #171717 !important;
    border: 1px solid #383838 !important;
    border-radius: 8px !important;
    color: #f5f5f5 !important;
}

textarea::placeholder {
    color: #737373 !important;
}

label span, h2, h3, h4 {
    color: #f5f5f5 !important;
    font-weight: 600 !important;
}

/* Primary Action Button */
button.primary-btn {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    height: 48px !important;
    margin-top: 10px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}

button.primary-btn:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
}
"""

with gr.Blocks(css=custom_css, title="Advanced ATS Analyzer") as demo:
    with gr.Column(elem_classes=["app-header"]):
        gr.Markdown("# Advanced ATS Analyzer")
        gr.Markdown("Semantic Matching • Skill Taxonomy • AI Suggestions")
    
    with gr.Row():
        # LEFT PANEL: INPUT CONFIGURATION
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            gr.Markdown("### Input Configuration")
            pdf_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
            jd_input = gr.Textbox(
                label="Job Specification", 
                lines=8, 
                placeholder="Paste target job description skills and requirements here..."
            )
            submit_btn = gr.Button("Execute Match Matrix", variant="primary", elem_classes=["primary-btn"])
            
        # RIGHT PANEL: INTELLIGENCE TELEMETRY / OUTPUTS
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            gr.Markdown("### Intelligence Telemetry")
            output_report = gr.Markdown(label="Skill Gap & Similarity Metrics")
            output_preview = gr.Textbox(
                label="Extracted Resume Stream", 
                lines=5, 
                interactive=False
            )
            
    submit_btn.click(
        fn=analyze_resume_comprehensive, 
        inputs=[pdf_input, jd_input], 
        outputs=[output_report, output_preview]
    )

if __name__ == "__main__":
    demo.launch()