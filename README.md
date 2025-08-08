📌 Overview
This project is an open-source candidate recommendation engine built with Streamlit.
It matches job descriptions with candidate resumes (.txt, .docx, .pdf) using semantic embeddings and cosine similarity, and generates both a detailed summary and a one-liner reason explaining why a candidate is suitable for the role.

The app runs entirely locally — no external APIs — and uses open-source models:

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Summarization: google/flan-t5-small

✨ Features
📂 Upload one or multiple resumes in .txt, .docx, or .pdf format.

🔍 Match candidates to a job description using semantic embeddings.

📊 Rank candidates by cosine similarity score.

📝 Detailed summary: 2–3 line explanation of the candidate’s fit.

💡 One-liner reason: a short, direct suitability statement.

⏱ Displays computation time for embeddings, similarity, and summary generation.

🖥 100% local processing — keeps data private.

Installation
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
2. Create a virtual environment
Windows (PowerShell):

powershell
Copy
Edit
python -m venv .venv
.venv\Scripts\Activate
macOS / Linux:

bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
bash
Copy
Edit
python -m pip install --upgrade pip
pip install -r requirements.txt
🚀 Running the App
bash
Copy
Edit
streamlit run app.py
Open the URL shown in the terminal (default: http://localhost:8501) in your browser.

📂 Usage
Paste the job description into the text area.

Upload one or more resumes (.txt, .docx, .pdf).

Select how many top candidates to display.

Click "Match Candidates".

View:

Cosine similarity score

Detailed summary

One-liner reason for suitability

Processing time breakdown

📐 How It Works
Step 1: The job description and resumes are converted into embeddings using all-MiniLM-L6-v2.

Step 2: Cosine similarity is calculated between the job description and each resume.

Step 3: Candidates are ranked by similarity score.

Step 4: Summaries and one-liners are generated using flan-t5-small.

Calculate Cosine Similarity 
 
Requirements
Python 3.9–3.11

Dependencies in requirements.txt:

streamlit

sentence-transformers

scikit-learn

transformers

torch

chardet

docx2txt

pdfplumber

