"""
Database models for RecruitBot.

This file defines the main database tables used for:
- Resume storage
- Job description storage
- Candidate-job match results
- Chatbot conversation logs
- HR user authentication
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Stores parsed resume details and extracted NLP features
class Resume(db.Model):
    __tablename__ = "resumes"

    id           = db.Column(db.Integer, primary_key=True)
    # Custom resume identifier (example: DS001)
    resume_id    = db.Column(db.String(64), unique=True, nullable=False)  
    raw_text     = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text)
    category     = db.Column(db.String(128))
    skills       = db.Column(db.Text)          # JSON string list
    experience   = db.Column(db.String(64))
    embedding    = db.Column(db.Text)          # JSON float list
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    match_results = db.relationship("MatchResult", back_populates="resume", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":         self.id,
            "resume_id":  self.resume_id,
            "category":   self.category,
            "skills":     self.skills,
            "experience": self.experience,
            "created_at": self.created_at.isoformat(),
        }

# Stores uploaded job descriptions and extracted job requirements
class JobDescription(db.Model):
    __tablename__ = "job_descriptions"

    id               = db.Column(db.Integer, primary_key=True)
    filename         = db.Column(db.String(256), nullable=False)
    raw_text         = db.Column(db.Text, nullable=False)
    cleaned_text     = db.Column(db.Text)
    category         = db.Column(db.String(128))       # predicted by ML model
    required_skills  = db.Column(db.Text)              # JSON list
    required_exp     = db.Column(db.String(64))
    embedding        = db.Column(db.Text)              # JSON float list
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)

    match_results = db.relationship(
        "MatchResult", back_populates="jd", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id":              self.id,
            "filename":        self.filename,
            "category":        self.category,
            "required_skills": self.required_skills,
            "required_exp":    self.required_exp,
            "created_at":      self.created_at.isoformat(),
        }

# Stores candidate-job matching scores and rankings
class MatchResult(db.Model):
    __tablename__ = "match_results"

    id               = db.Column(db.Integer, primary_key=True)
    jd_id            = db.Column(db.Integer, db.ForeignKey("job_descriptions.id"), nullable=False)
    resume_id        = db.Column(db.Integer, db.ForeignKey("resumes.id"), nullable=False)
    final_score      = db.Column(db.Float, default=0.0)
    skill_score      = db.Column(db.Float, default=0.0)
    experience_score = db.Column(db.Float, default=0.0)
    education_score  = db.Column(db.Float, default=0.0)
    semantic_score   = db.Column(db.Float, default=0.0)
    rank             = db.Column(db.Integer, default=0)
    matched_skills   = db.Column(db.Text)
    missing_skills   = db.Column(db.Text)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)

    jd     = db.relationship("JobDescription", back_populates="match_results")
    resume = db.relationship("Resume",         back_populates="match_results")

    def to_dict(self):
        return {
            "id":               self.id,
            "jd_id":            self.jd_id,
            "resume_id":        self.resume_id,
            "dataset_resume_id":self.resume.resume_id  if self.resume else "",
            "category":         self.resume.category   if self.resume else "",
            "final_score":      round(self.final_score      * 100, 1),
            "skill_score":      round(self.skill_score      * 100, 1),
            "experience_score": round(self.experience_score * 100, 1),
            "education_score":  round(self.education_score  * 100, 1),
            "semantic_score":   round(self.semantic_score   * 100, 1),
            "rank":             self.rank,
            "matched_skills":   self.matched_skills,
            "missing_skills":   self.missing_skills,
            "experience":       self.resume.experience if self.resume else "",
        }

# Stores chatbot conversation history linked to selected job descriptions
class ChatLog(db.Model):
    __tablename__ = "chat_logs"

    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(128))
    role       = db.Column(db.String(16))
    message    = db.Column(db.Text)
    jd_id      = db.Column(db.Integer, db.ForeignKey("job_descriptions.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Stores HR login credentials for dashboard access
class HRUser(db.Model):
    __tablename__ = "hr_users"

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(100), nullable=False)  # plain text (for demo only)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "password": self.password,  # optional (you can remove later)
            "created_at": self.created_at.isoformat()
        }