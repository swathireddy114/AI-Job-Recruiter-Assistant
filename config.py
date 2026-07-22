import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "development-secret-key")

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:your_password@127.0.0.1:3306/db_recruiter_assistant"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

    MODEL_PATH = os.path.join(BASE_DIR, "models_pkl", "lr_model.pkl")
    VECTORIZER_PATH = os.path.join(
        BASE_DIR,
        "models_pkl",
        "tfidf_vectorizer.pkl"
    )

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")