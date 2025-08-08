 Overview
This project is an open-source candidate recommendation engine built with Streamlit.
It matches job descriptions with candidate resumes (.txt, .docx, .pdf) using semantic embeddings and cosine similarity, and generates both a detailed summary and a one-liner reason explaining why a candidate is suitable for the role.
The app runs entirely locally — no external APIs — and uses open-source models:
•	Embeddings: sentence-transformers/all-MiniLM-L6-v2
•	Summarization: google/flan-t5-small
________________________________________
✨ Features
•	Upload one or multiple resumes in .txt, .docx, or .pdf format.
•	Match candidates to a job description using semantic embeddings.
•	Rank candidates by cosine similarity score.
•	Detailed summary: 2–3 line explanation of the candidate’s fit.
•	 One-liner reason: a short, direct suitability statement.
•	Displays computation time for embeddings, similarity, and summary generation.
•	100% local processing — keeps data private.
________________________________________
📦 Installation
1. Clone the repository
2. Create a virtual environment
Windows (PowerShell):
powershell
python -m venv .venv
.venv\Scripts\Activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
________________________________________
Running the App
streamlit run app.py
Open the URL shown in the terminal (default: http://localhost:8501) in your browser.
________________________________________
📂 Usage
1.	Paste the job description into the text area.
2.	Upload one or more resumes (.txt, .docx, .pdf).
3.	Select how many top candidates to display.
4.	Click "Match Candidates".
5.	View:
o	Cosine similarity score
o	Detailed summary
o	One-liner reason for suitability
o	Processing time breakdown
________________________________________
📐 How It Works
•	Step 1: The job description and resumes are converted into embeddings using all-MiniLM-L6-v2.
•	Step 2: Cosine similarity is calculated between the job description and each resume.
•	Step 3: Candidates are ranked by similarity score.
•	Step 4: Summaries and one-liners are generated using flan-t5-small.
Cosine Similarity Formula:
cos⁡(θ)=A⋅B∥A∥⋅∥B∥\cos(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|}cos(θ)=∥A∥⋅∥B∥A⋅B 
________________________________________
🛠 Requirements
•	Python 3.9–3.11
•	Dependencies in requirements.txt:
o	streamlit
o	sentence-transformers
o	scikit-learn
o	transformers
o	torch
o	chardet
o	docx2txt
o	pdfplumber

