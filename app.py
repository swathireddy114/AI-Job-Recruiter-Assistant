"""
RecruitBot AI Recruitment System

This Flask application provides:
- Resume upload and parsing
- Job description upload and classification
- AI-powered candidate matching
- Resume ranking dashboard
- HR chatbot assistant using Groq LLM

Technologies:
Flask, SQLAlchemy, Sentence Transformers, LangChain, Groq API
"""

import os
import json
import uuid
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, flash
)

from config import Config
from models import db, Resume, JobDescription, MatchResult, ChatLog, HRUser
import inference

# Flask application setup
app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

def from_json_filter(value):
    """Parse a JSON string safely; return list on failure."""
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


app.jinja_env.filters["from_json"] = from_json_filter

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load trained classification model and TF-IDF vectorizer
models_loaded = inference.load_models(
    app.config["MODEL_PATH"],
    app.config["VECTORIZER_PATH"],
)

# Utility helper functions for file validation and session management
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def get_or_create_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


@app.route("/")
def login_page():    
    return render_template("login.html")



@app.route("/logout")
def logout():
    session.pop("hr_user", None)   # remove logged-in user
    return render_template("login.html")

@app.route("/index")
def index():
    if "hr_user" not in session:
        return redirect(url_for("login_page"))
    total_resumes = Resume.query.count()
    total_jds = JobDescription.query.count()
    total_matches = MatchResult.query.count()
    recent_jds = JobDescription.query.order_by(JobDescription.created_at.desc()).limit(5).all()
    return render_template(
        "index.html",
        total_resumes=total_resumes,
        total_jds=total_jds,
        total_matches=total_matches,
        recent_jds=recent_jds,
        models_loaded=models_loaded,
    )
    
# Handles HR login authentication
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    # Retrieve HR user by email before checking password
    hr = HRUser.query.filter_by(email=email).first()

    if not hr:
        flash("Invalid email address")
        return redirect(url_for("login_page"))

    # Password comparison is case-sensitive
    if hr.password == password:
        session["hr_user"] = hr.id
        return redirect("/index")
    else:
        flash("Wrong password")
        return redirect(url_for("login_page"))


@app.route("/upload-resume", methods=["GET", "POST"])
def upload_resume():
    print("\n==== /upload-resume called ====")

    if request.method == "GET":
        print("GET request → loading resumes")
        resumes = Resume.query.order_by(Resume.created_at.desc()).all()
        return render_template("upload_resume.html", resumes=resumes)

    # Debug uploaded request data
    print("Request method:", request.method)
    print("request.files:", request.files)
    print("request.form:", request.form)

    # Collect uploaded resume files
    files = request.files.getlist("resume_files")
    print("Files from resume_files:", files)

    if not files or files == [None]:
        single_file = request.files.get("resume")
        print("Single file (resume):", single_file)
        if single_file:
            files = [single_file]

    print("Final files list:", files)

    results = []

    # Process each uploaded resume file
    for file in files:
        print("\n--- Processing file ---")

        if not file:
            print("File is None → skipping")
            continue

        print("Filename:", file.filename)

        if file.filename == "":
            print("Empty filename → skipping")
            continue

        if not allowed_file(file.filename):
            print("File type not allowed:", file.filename)
            continue

        try:
            fname = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)

            print("Saving file to:", save_path)
            file.save(save_path)

            # Generate resume identifier
            stem = os.path.splitext(fname)[0]
            resume_id = stem

            custom_id = request.form.get("resume_id", "").strip()
            if custom_id:
                resume_id = custom_id

            print("Resume ID:", resume_id)

            # Skip duplicate resume records
            if Resume.query.filter_by(resume_id=resume_id).first():
                print("Duplicate resume → skipping")
                results.append({
                    "file": fname,
                    "status": "skipped",
                    "reason": f"resume_id '{resume_id}' already exists"
                })
                continue

            # Extract raw text from uploaded resume
            try:
                raw_text = inference.load_file(save_path)
                print("Raw text length:", len(raw_text))
            except Exception as e:
                print(" FILE READ ERROR:", e)
                raise e

            # Extract resume features using NLP pipeline
            try:
                cleaned  = inference.clean_text(raw_text)
                category = inference.predict_category(raw_text)
                skills   = inference.extract_skills(raw_text)
                exp      = inference.extract_experience(raw_text)
                emb      = inference.get_embedding(cleaned)

                print("Category:", category)
                print("Skills:", skills)
                print("Experience:", exp)
                print("Embedding length:", len(emb) if emb else 0)

            except Exception as e:
                print(" PROCESSING ERROR:", e)
                raise e

            # Save processed resume details to database
            resume = Resume(
                resume_id=resume_id,
                raw_text=raw_text[:20000],
                cleaned_text=cleaned[:20000],
                category=category,
                skills=json.dumps(skills),
                experience=exp,
                embedding=json.dumps(emb) if emb else "[]"
            )

            db.session.add(resume)
            db.session.commit()
            print(" Resume saved to DB")

            # Match uploaded resume against existing job descriptions
            try:
                all_jds = JobDescription.query.all()
                print("Total JDs:", len(all_jds))

                for jd in all_jds:
                    jd_emb = json.loads(jd.embedding or "[]")
                    jd_skills = json.loads(jd.required_skills or "[]")

                    scores = inference.score_match(
                        jd_skills=jd_skills,
                        resume_skills=skills,
                        jd_exp=jd.required_exp or "",
                        resume_exp=exp or "",
                        jd_emb=jd_emb,
                        resume_emb=emb,
                    )

                    print("Match score:", scores["final_score"])

                    mr = MatchResult(
                        jd_id=jd.id,
                        resume_id=resume.id,
                        final_score=scores["final_score"],
                        skill_score=scores["skill_score"],
                        experience_score=scores["experience_score"],
                        education_score=scores["education_score"],
                        semantic_score=scores["semantic_score"],
                        matched_skills=scores["matched_skills"],
                        missing_skills=scores["missing_skills"],
                    )

                    db.session.add(mr)

                db.session.commit()
                print(" Matching completed")

            except Exception as e:
                print(" MATCHING ERROR:", e)
                raise e

            # Update candidate ranking for each job description
            try:
                for jd in all_jds:
                    matches = MatchResult.query.filter_by(jd_id=jd.id)\
                        .order_by(MatchResult.final_score.desc()).all()

                    for rank, m in enumerate(matches, start=1):
                        m.rank = rank

                db.session.commit()
                print(" Ranking updated")

            except Exception as e:
                print(" RANKING ERROR:", e)
                raise e

            results.append({
                "file": fname,
                "status": "success",
                "resume_id": resume_id,
                "category": category
            })

        except Exception as e:
            import traceback
            traceback.print_exc()

            results.append({
                "file": file.filename,
                "status": "error",
                "error": str(e)
            })

    print("Final Results:", results)
    return jsonify({"results": results})

@app.route("/upload-jd", methods=["GET", "POST"])
def upload_jd():
    if request.method == "GET":
        jds = JobDescription.query.order_by(JobDescription.created_at.desc()).all()
        return render_template("upload_jd.html", jds=jds)

    file = request.files.get("jd_file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    fname     = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(save_path)

    try:
        print("Inside upload Try")
        raw_text = inference.load_file(save_path)
        cleaned  = inference.clean_text(raw_text)

        # Predict JD category using trained ML model
        category = inference.predict_category(raw_text)
        print("Prediction", category)

        skills   = inference.extract_skills(raw_text)
        exp      = inference.extract_experience(raw_text)
        emb      = inference.get_embedding(cleaned)

        jd = JobDescription(
            filename        = fname,
            raw_text        = raw_text[:20000],
            cleaned_text    = cleaned[:20000],
            category        = category,
            required_skills = json.dumps(skills),
            required_exp    = exp,
            embedding = json.dumps(emb) if emb else "[]"
        )
        db.session.add(jd)
        db.session.commit()
        print("commited")
        
        # Match JD with resumes from the same category; fall back to all resumes if needed
        candidates = Resume.query.filter_by(category=category).all()
        if len(candidates) < 3:
            candidates = Resume.query.all()

        jd_emb         = json.loads(jd.embedding or "[]")
        jd_skills_list = json.loads(jd.required_skills or "[]")

        match_rows = []
        for res in candidates:
            res_skills = json.loads(res.skills    or "[]")
            res_emb    = json.loads(res.embedding or "[]")

            scores = inference.score_match(
                jd_skills    = jd_skills_list,
                resume_skills= res_skills,
                jd_exp       = jd.required_exp or "",
                resume_exp   = res.experience  or "",
                jd_emb       = jd_emb,
                resume_emb   = res_emb,
            )
            mr = MatchResult(
                jd_id            = jd.id,
                resume_id        = res.id,
                final_score      = scores["final_score"],
                skill_score      = scores["skill_score"],
                experience_score = scores["experience_score"],
                education_score  = scores["education_score"],
                semantic_score   = scores["semantic_score"],
                matched_skills   = scores["matched_skills"],
                missing_skills   = scores["missing_skills"],
            )
            match_rows.append(mr)

        # Rank by final_score descending and keep only top 5
        match_rows.sort(key=lambda x: x.final_score, reverse=True)
        top_5_matches = match_rows[:5]  # Only keep top 5
        print(top_5_matches)
        
        for rank, mr in enumerate(top_5_matches, start=1):
            mr.rank = rank
            db.session.add(mr)
        
        db.session.commit()

        return jsonify({
            "jd_id":         jd.id,
            "category":      category,
            "matched_count": len(top_5_matches),  # Will always be up to 5
            "redirect":      url_for("dashboard", jd_id=jd.id),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# Displays ranked candidate matches for a selected JD
@app.route("/dashboard")
def dashboard():
    if "hr_user" not in session:
        return redirect(url_for("login_page"))

    jd_id = request.args.get("jd_id", type=int)
    jds   = JobDescription.query.order_by(JobDescription.created_at.desc()).all()
    selected_jd = None
    matches = []

    if jd_id:
        selected_jd = JobDescription.query.get_or_404(jd_id)
        matches = (
            MatchResult.query
            .filter_by(jd_id=jd_id)
            .order_by(MatchResult.rank)
            .limit(20)
            .all()
        )

    return render_template(
        "dashboard.html",
        jds=jds,
        selected_jd=selected_jd,
        matches=matches,
    )

# Displays RecruitBot chatbot interface
@app.route("/chatbot")
def chatbot():
    if "hr_user" not in session:
        return redirect(url_for("login_page"))
    jd_id = request.args.get("jd_id", type=int)
    selected_jd = None
    if jd_id:
        selected_jd = JobDescription.query.get(jd_id)
    jds = JobDescription.query.order_by(JobDescription.created_at.desc()).all()
    return render_template("chatbot.html", selected_jd=selected_jd, jds=jds)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    import os

    data = request.get_json()
    user_msg  = data.get("message", "").strip()
    jd_id     = data.get("jd_id")
    history   = data.get("history", [])  # list of {role, content}
    sid       = get_or_create_session()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Build chatbot context from selected JD and candidate ranking data
    context_parts = []
    if jd_id:
        jd = JobDescription.query.get(jd_id)
        if jd:
            top_matches = (
                MatchResult.query
                .filter_by(jd_id=jd_id)
                .order_by(MatchResult.rank)
                .limit(10)
                .all()
            )
            jd_info = (
                f"Category: {jd.category}\n"
                f"Required Skills: {jd.required_skills}\n"
                f"Required Experience: {jd.required_exp}\n"
            )
            candidates_info = "\n".join(
                f"Rank {m.rank}: {m.resume.category if m.resume else 'Unknown'} | "
                f"Score: {round(m.final_score*100,1)}% | "
                f"Skills: {round(m.skill_score*100,1)}% | "
                f"Exp: {round(m.experience_score*100,1)}% | "
                f"Edu: {round(m.education_score*100,1)}% | "
                f"Matched: {m.matched_skills} | Missing: {m.missing_skills}"
                for m in top_matches
            )
            context_parts.append(jd_info)
            context_parts.append("TOP CANDIDATES:\n" + candidates_info)

    system_prompt = f"""You are RecruitBot, an expert AI recruiter assistant embedded in a 
recruitment platform. You help HR professionals understand candidate rankings, explain 
match scores, and answer recruitment questions with clarity and insight.

Be conversational, professional, and helpful. Use specific data when available.
When explaining rankings, mention specific skills, experience gaps, and education fit.
Don't give emoji in response
Current Context:
{chr(10).join(context_parts) if context_parts else "No job description selected yet. Answer general recruitment questions."}
"""

    # Build messages for Groq (langchain format)
    messages = []
    # Add system message first
    messages.append(SystemMessage(content=system_prompt))
    
    # Add conversation history
    for h in history[-10:]:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            messages.append(AIMessage(content=h["content"]))
    
    # Add current user message
    messages.append(HumanMessage(content=user_msg))

    # Initialize Groq LLM for chatbot response generation
    groq_api_key = app.config["GROQ_API_KEY"]
    groq_api_key = app.config.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not groq_api_key:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b",  # or any model you have access to
            temperature=0.0,
            max_tokens=800
        )
        print("+++++++++++++++Printing Message+++++++++++++++++++++++++")
        print(messages)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # Invoke the model
        response = llm.invoke(messages)
        bot_reply = response.content

    except Exception as e:
        bot_reply = f"I'm having trouble connecting right now. Error: {str(e)}"

    # Store chatbot conversation in database
    try:
        db.session.add(ChatLog(session_id=sid, role="user", message=user_msg, jd_id=jd_id))
        db.session.add(ChatLog(session_id=sid, role="assistant", message=bot_reply, jd_id=jd_id))
        db.session.commit()
    except Exception:
        pass

    return jsonify({"reply": bot_reply})


# API routes for resume and match data
@app.route("/api/resumes")
def api_resumes():
    resumes = Resume.query.order_by(Resume.created_at.desc()).all()
    return jsonify([r.to_dict() for r in resumes])


@app.route("/api/resume/<int:resume_id>")
def api_resume_detail(resume_id):
    r = Resume.query.get_or_404(resume_id)
    return jsonify(r.to_dict())


@app.route("/api/jd/<int:jd_id>/matches")
def api_jd_matches(jd_id):
    matches = (
        MatchResult.query
        .filter_by(jd_id=jd_id)
        .order_by(MatchResult.rank)
        .limit(20)
        .all()
    )
    return jsonify([m.to_dict() for m in matches])


@app.route("/api/delete-resume/<int:resume_id>", methods=["DELETE"])
def delete_resume(resume_id):
    r = Resume.query.get_or_404(resume_id)
    db.session.delete(r)
    db.session.commit()
    return jsonify({"status": "deleted"})

@app.route("/api/delete-jd/<int:jd_id>", methods=["DELETE"])
def delete_jd(jd_id):
    jd = JobDescription.query.get_or_404(jd_id)

    # Delete related chat logs first
    ChatLog.query.filter_by(jd_id=jd_id).delete()
    MatchResult.query.filter_by(jd_id=jd_id).delete()

    db.session.delete(jd)
    db.session.commit()

    return jsonify({"status": "deleted"})

from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


@app.route("/download-resume/<int:resume_id>")
def download_resume(resume_id):
    # Retrieve resume from database
    resume = Resume.query.get(resume_id)

    if not resume:
        return "Resume not found", 404

    # Generate downloadable PDF in memory
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    # Split text into paragraphs
    for line in resume.raw_text.split("\n"):
        if line.strip():
            content.append(Paragraph(line.strip(), styles["Normal"]))
            content.append(Spacer(1, 10))

    doc.build(content)

    buffer.seek(0)

    # Return generated PDF as downloadable file
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{resume.resume_id}.pdf",
        mimetype="application/pdf"
    )

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
