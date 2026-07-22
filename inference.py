"""
Inference and NLP utilities for RecruitBot.

This module handles:
- Loading trained ML models
- Reading PDF, DOCX, and TXT files
- Cleaning and preprocessing text
- Extracting skills, experience, and education
- Generating Sentence-BERT embeddings
- Calculating candidate-job match scores
"""

import re
import json
import os
import math
import joblib
import contractions
from textblob import TextBlob
import nltk

# Download required NLTK resources if missing
for pkg in ["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words("english"))

# Resume category labels used by the trained classifier
CLASS_LABELS = [
    "Advocate", "Arts", "Automation Testing", "Blockchain", "Business Analyst",
    "Civil Engineer", "Data Science", "Database", "DevOps Engineer",
    "DotNet Developer", "ETL Developer", "Electrical Engineering", "HR",
    "Hadoop", "Health and fitness", "Java Developer", "Mechanical Engineer",
    "Network Security Engineer", "Operations Manager", "PMO",
    "Python Developer", "SAP Developer", "Sales", "Testing", "Web Designing",
]

# Skill keyword list used for rule-based skill extraction
SKILL_KEYWORDS = [
    "python", "java", "c", "c++", "c#", "dotnet", ".net", "asp.net", "mvc",
    "javascript", "typescript", "html", "css", "bootstrap", "react", "angular", "vue",
    "node", "node.js", "express", "django", "flask",

    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "pl/sql",
    "database", "dbms", "etl", "data warehousing", "data modeling",

    "machine learning", "deep learning", "nlp", "data science", "data analysis",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "xgboost", "matplotlib", "seaborn", "LangChain", "RAG",

    "hadoop", "spark", "hive", "pig", "kafka", "big data",

    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "ci/cd", "devops", "linux", "shell scripting",

    "selenium", "automation testing", "manual testing", "testng", "junit",
    "pytest", "quality assurance", "qa",

    "blockchain", "solidity", "web3", "cryptography",

    "networking", "network security", "cybersecurity", "penetration testing",
    "ethical hacking", "firewall",

    "sap", "sap abap", "sap hana", "sap fico", "erp",

    "business analysis", "business analyst", "requirement gathering",
    "stakeholder management", "use cases",

    "project management", "pmo", "agile", "scrum", "kanban",

    "hr", "recruitment", "payroll", "employee engagement", "talent acquisition",

    "sales", "crm", "lead generation", "negotiation", "marketing",

    "operations", "operations management", "supply chain", "logistics",

    "civil", "autocad", "revit", "construction", "structural design",
    "mechanical", "cad", "cam", "thermodynamics",
    "electrical", "power systems", "circuit design",

    "health", "fitness", "nutrition", "yoga",

    "web designing", "ui", "ux", "photoshop", "illustrator", "figma",

    "advocate", "legal research", "litigation", "contract law",

    "communication", "leadership", "teamwork", "problem solving",
    "time management", "critical thinking"
]


_model = None
_vectorizer = None

def load_models(model_path: str, vectorizer_path: str):
    global _model, _vectorizer
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vectorizer_path)
        return True
    return False


# File reading helpers for PDF, DOCX, and TXT resumes/JDs
def read_pdf(file_path: str) -> str:
    try:
        import fitz
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"[PDF read error: {e}]"


def read_docx(file_path: str) -> str:
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[DOCX read error: {e}]"


def read_txt(file_path: str) -> str:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def load_file(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return read_pdf(file_path)
    elif ext == "docx":
        return read_docx(file_path)
    elif ext == "txt":
        return read_txt(file_path)
    raise ValueError(f"Unsupported file format: {ext}")


# Text preprocessing used before classification and embedding generation
def clean_text(text: str) -> str:
    text = contractions.fix(text)
    emoticons = [r":\)", r":\(", r":P"]
    text = re.sub("|".join(emoticons), "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        blob = TextBlob(text)
        lemmas = [w.lemmatize() for w in blob.words if w.lower() not in STOP_WORDS]
        text = " ".join(lemmas).lower()
    except Exception:
        text = text.lower()
    return text


def predict_category(text: str) -> str:
    """Predict category from raw text."""
    if _model is None or _vectorizer is None:
        return "Unknown"
    cleaned = clean_text(text)
    vec     = _vectorizer.transform([cleaned])
    pred    = _model.predict(vec)[0]

    # Model may return: int index, float index, or string label directly
    # Case 1: numeric index  →  e.g. 6  or  "6"
    try:
        idx = int(float(str(pred)))
        if 0 <= idx < len(CLASS_LABELS):
            return CLASS_LABELS[idx]
    except (ValueError, TypeError):
        pass

    # Case 2: already a string label  →  e.g. "Data Science"
    pred_str = str(pred).strip()
    if pred_str in CLASS_LABELS:
        return pred_str

    # Case 3: partial match (case-insensitive)
    pred_lower = pred_str.lower()
    for label in CLASS_LABELS:
        if label.lower() == pred_lower:
            return label

    return pred_str   


def predict_category_from_file(file_path: str) -> str:
    text = load_file(file_path)
    return predict_category(text)


# Rule-based skill extraction from resume or JD text
def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = [s for s in SKILL_KEYWORDS if re.search(r"\b" + re.escape(s) + r"\b", text_lower)]
    return list(dict.fromkeys(found))  # preserve order, deduplicate


# Regex-based experience extraction
def extract_experience(text: str) -> str:
    patterns = [
        r"(\d+\+?\s*(?:to\s*\d+)?\s*years?\s*(?:of\s*)?(?:experience|exp)?)",
        r"(experience\s*:?\s*\d+\+?\s*years?)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return "Not specified"


# Simple education keyword extraction
EDU_KEYWORDS = [
    "phd", "ph.d", "doctorate", "master", "m.tech", "mba", "m.sc", "m.s",
    "bachelor", "b.tech", "b.e", "b.sc", "b.s", "diploma", "undergraduate",
    "postgraduate", "degree",
]

def extract_education(text: str) -> str:
    text_lower = text.lower()
    for kw in EDU_KEYWORDS:
        if kw in text_lower:
            # grab surrounding context
            idx = text_lower.find(kw)
            snippet = text[max(0, idx - 10): idx + 60].strip()
            return snippet
    return "Not specified"

from sentence_transformers import SentenceTransformer

# Load Sentence-BERT model once to avoid repeated loading
_st_model = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1")

def get_embedding(text: str) -> list:
    try:
        if not text:
            return []

        # Convert text → semantic vector (captures meaning, not just keywords)
        vec = _st_model.encode(text)
        return vec.tolist()

    except Exception as e:
        print("Embedding Error:", e)
        return []


def cosine_similarity_vectors(v1: list, v2: list) -> float:
    # Measures semantic similarity between JD and Resume embeddings

    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))

    if n1 == 0 or n2 == 0:
        return 0.0

    return dot / (n1 * n2)   # 1 → very similar, 0 → unrelated


def score_match(jd_skills: list, resume_skills: list,
                jd_exp: str,    resume_exp: str,
                jd_emb: list,   resume_emb: list) -> dict:
    """
    Core matching logic:
    Combines skill match + experience + semantic similarity
    """

    # 1. Skill match score 
    jd_set  = set(s.lower() for s in jd_skills)
    res_set = set(s.lower() for s in resume_skills)

    matched = list(jd_set & res_set)   # common skills
    missing = list(jd_set - res_set)   # required but not present

    skill_score = len(matched) / max((len(jd_set) + 0.5 * len(missing)), 1)
    

    # 2. Experience match score 
    def parse_years(exp_str: str) -> float:
        # Extract numeric value from text like "3 years"
        nums = re.findall(r"\d+", exp_str or "")
        return float(nums[0]) if nums else 0.0

    jd_yrs  = parse_years(jd_exp)
    res_yrs = parse_years(resume_exp)

    if jd_yrs == 0:
        exp_score = 1.0
    else:
        exp_score = min(res_yrs / jd_yrs, 1.0)


    # 3. Semantic similarity score
    if jd_emb and resume_emb:
        sem_score = cosine_similarity_vectors(jd_emb, resume_emb)
    else:
        sem_score = 0.5   # fallback if embeddings missing


     # 4. Final weighted score
    # Skills (50%) + Experience (20%) + Semantic (30%)
    final = (
        0.5 * skill_score +
        0.2 * exp_score +
        0.3 * sem_score
    )

    return {
        "skill_score":      round(skill_score, 4),
        "experience_score": round(exp_score,   4),
        "education_score":  1.0,   # kept for schema compatibility
        "semantic_score":   round(sem_score,   4),
        "final_score":      round(final,       4),
        "matched_skills":   json.dumps(matched),
        "missing_skills":   json.dumps(missing),
    }