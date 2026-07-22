# A Job Recruiter Assistant Chatbot

An AI-powered recruitment assistant system that automates resume analysis, job description understanding, semantic candidate matching, and recruiter support using Machine Learning, NLP, Sentence-BERT embeddings, and Generative AI chatbot integration.

---

# Project Overview

Traditional recruitment processes involve manual resume screening, keyword-based filtering, and repetitive candidate evaluation, which are time-consuming and inefficient. This project addresses these limitations by developing an intelligent recruitment assistant capable of:

- Parsing resumes automatically
- Extracting skills and experience
- Understanding job descriptions
- Performing semantic similarity matching
- Ranking candidates intelligently
- Providing explainable AI-based recruiter assistance through a chatbot

The system uses Sentence-BERT embeddings and cosine similarity to perform contextual candidate matching beyond simple keyword overlap.

---

# Features

## Resume Processing
- Upload resumes in PDF, DOCX, and TXT formats
- Automatic text extraction and preprocessing
- Skill extraction using predefined NLP skill dictionary
- Experience extraction using regex-based parsing
- Resume category prediction using Machine Learning classification

## Job Description Processing
- Upload and analyze job descriptions
- Extract required skills and experience
- Generate semantic embeddings

## Intelligent Candidate Matching
- Skill matching
- Experience matching
- Semantic similarity matching using Sentence-BERT
- Weighted final scoring system
- Automatic candidate ranking

## Recruiter Dashboard
- View ranked candidates
- Matched and missing skill visualization
- Match score breakdown
- Resume download functionality
- Dynamic re-ranking when new resumes are uploaded

## AI Recruiter Chatbot
- Explain candidate rankings
- Answer recruiter questions
- Provide transparent reasoning for match scores
- Markdown-rendered responses
- Voice input support (Speech-to-Text)

---

# Technologies Used

| Category | Technologies |
|---|---|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Python, Flask |
| Database | MySQL |
| Machine Learning | Scikit-learn |
| NLP | Sentence-Transformers, Regex |
| Deep Learning | Sentence-BERT |
| Semantic Matching | Cosine Similarity |
| Data Handling | Pandas, NumPy |
| Authentication | Flask Sessions |
| Development Environment | Anaconda, Jupyter Notebook, VS Code |

---

# Project Structure

```bash
A-Job-Recruiter-Assistant-Chatbot/
│
├── app/
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── chatbot.html
│   │   ├── upload_resume.html
│   │   ├── upload_jd.html
│   │   ├── login.html
│   │   ├── index.html
│   │
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── uploads/
│   │
│   ├── app.py
│   ├── inference.py
│   ├── models.py
│   ├── config.py
│   ├── requirements.txt
│
├── models_pkl/
│   ├── lr_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── db_recruiter_assistant.sql
├── README.md
```

---

# Installation and Setup

## 1. Create Conda Environment

```bash
conda create -n recruiter_ai python=3.10
```

Activate the environment:

```bash
conda activate recruiter_ai
```

---

## 2. Navigate to Project Directory

```bash
cd path_to_project_folder
```

Example:

```bash
cd C:\Users\Admin\Desktop\A-Job-Recruiter-Assistant-Chatbot
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# API Configuration

Open:

```text
config.py
```

Add your Groq API key:

```python
GROQ_API_KEY = "your_api_key_here"
```

---

# Database Setup

## 1. Create Database

Open MySQL Workbench and run:

```sql
CREATE DATABASE db_recruiter_assistant;
```

---

## 2. Import Database File

Import the provided SQL file:

```text
db_recruiter_assistant.sql
```

into the created database.

---

## 3. Configure Database Connection

Open:

```text
config.py
```

Update:

```python
SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:your_password@localhost:3306/db_recruiter_assistant"
```

---

# Running the Application

## Start Flask Server

```bash
python app.py
```

Server runs on:

```text
http://127.0.0.1:5000
```

---

# HR Login Credentials

Use the following credentials to access the recruiter dashboard:

```text
Email: hr@gmail.com
Password: Admin@123
```

---

# Candidate Matching Workflow

```text
Job Description
      ↓
Skill Extraction
      ↓
Experience Extraction
      ↓
Sentence-BERT Embedding
      ↓
Resume Processing
      ↓
Skill Match + Experience Match + Semantic Similarity
      ↓
Weighted Score Calculation
      ↓
Final Candidate Ranking
      ↓
Recruiter Chatbot Explanation
```

---

# Scoring Formula

## Final Match Score

```text
Final Score =
0.5 × Skill Score +
0.2 × Experience Score +
0.3 × Semantic Score
```

---

# Semantic Similarity

Cosine similarity is used between Sentence-BERT embeddings:

```text
Similarity =
(A · B) / (||A|| × ||B||)
```

---

# System Modules

- Resume Upload Module
- Job Description Upload Module
- Candidate Ranking Dashboard
- Recruiter AI Chatbot
- Resume Download System
- Voice Interaction Module

---

# Privacy and Ethical Considerations

- The system focuses primarily on skills, experience, and semantic relevance during candidate matching.
- Personal candidate information is not directly used in semantic ranking calculations.
- The chatbot provides explainable responses for candidate ranking decisions.
- Human recruiter review is still required before final hiring decisions.

---

# Future Improvements

- Real-time ATS integration
- Bias mitigation techniques
- Cloud deployment
- Multi-language resume processing
- Advanced recruiter analytics
- Personalized candidate recommendations

---

# Author

## Swathi Venkatesh Reddy

MSc Advanced Computer Science  
University of Leicester

Focused on AI, NLP, Generative AI, and Intelligent Recruitment Systems.

---

# License

This project is developed for academic and research purposes.

---

# Acknowledgement

Special thanks to:

- Sentence-Transformers
- Scikit-learn
- Flask
- Hugging Face
- Groq API
- Open Source AI Community

---