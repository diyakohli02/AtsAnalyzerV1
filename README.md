# Advanced ATS Analyzer (Version 1)

An intelligent, lightweight web-based resume screening and skill gap analysis tool designed to bridge the gap between job seekers and Applicant Tracking Systems (ATS). It evaluates PDF resumes against target job descriptions using semantic text similarity and categorized skill taxonomy matching.

**Live Application:** [View Live Streamlit App](https://atsanalyzerv1git-jltrfdkztqurfgtdyjc4vq.streamlit.app/)[cite: 1]

---

## Tech Stack & Architecture

| Component | Technology |
| :--- | :--- |
| **Frontend / UI** | Streamlit (with Custom CSS Charcoal Dark Theme) |
| **PDF Parsing** | `pdfplumber` |
| **Semantic Matching** | `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`) |
| **Skill Taxonomy Engine** | Python RegEx (`re`) & Categorized Dictionaries |
| **Cloud Hosting** | Streamlit Community Cloud[cite: 1] |

---

## Why V1 Was Built This Way (The Engineering Pivot)
During deployment, heavy deep learning transformers (`sentence-transformers`, PyTorch) caused out-of-memory (OOM) crashes and thread freezing on cloud hosting free tiers due to massive binary weights. 
* **The Pivot:** We re-engineered the backend pipeline to use lightweight, high-performance `scikit-learn` TF-IDF vectorization and `pdfplumber`. This ensures instantaneous execution, zero memory bloat, and frictionless cloud deployment.

---

## Core Features
* **Instant Semantic Scoring:** Calculates cosine similarity between resume text and job descriptions.
* **Categorized Skill Taxonomy:** Breaks down requirements into Languages, Frameworks & Web, Databases, and Tools & DevOps.
* **Matched vs. Missing Telemetry:** Instantly highlights what skills you have and what keywords are missing.
* **Actionable Recommendations:** Provides tailored optimization tips to improve resume ATS performance.

---

## How to Run Locally

If you want to run or test the code on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/diyakohli02/AtsAnalyzerV1.git](https://github.com/diyakohli02/AtsAnalyzerV1.git)
   cd AtsAnalyzerV1
