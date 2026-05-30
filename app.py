import os
from flask import Flask, request, jsonify, render_template, session, Response
from werkzeug.utils import secure_filename
import pdfplumber
from claude_service import analyze_cv
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-prod")
app.config["TEMPLATES_AUTO_RELOAD"] = True

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/")
def index():
    return render_template(
        "index.html",
        cv_on_file=bool(session.get("cv_filename")),
        cv_name=session.get("cv_filename", ""),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    cv_file = request.files.get("cv")
    filepath = None

    if cv_file and cv_file.filename and _allowed_file(cv_file.filename):
        filename = secure_filename(cv_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv_file.save(filepath)
        session["cv_filename"] = filename
    elif session.get("cv_filename"):
        filename = session["cv_filename"]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            session.pop("cv_filename", None)
            return jsonify({"error": "Saved CV not found — please re-upload."}), 400
    else:
        return jsonify({"error": "Please upload a PDF CV."}), 400

    cv_text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    cv_text += text + "\n"
    except Exception as exc:
        return jsonify({"error": f"Failed to read PDF: {exc}"}), 400

    if not cv_text.strip():
        return jsonify(
            {"error": "No text found in the PDF. Ensure it contains selectable text (not a scanned image)."}
        ), 400

    try:
        result = analyze_cv(cv_text, job_description)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
