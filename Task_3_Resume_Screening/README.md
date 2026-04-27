# Resume / Candidate Screening System

This project is a beginner-friendly, real-world resume screening system that:
- extracts skills from resumes and a job description
- scores resumes by skill match and text similarity
- ranks candidates and highlights skill gaps

## What This System Does
- **Skill extraction** using a curated skill list (`data/skills.json`)
- **Job parsing** with required vs nice-to-have separation
- **Similarity scoring** using TF-IDF + cosine similarity
- **Skill gap analysis** to show missing required skills
- **Candidate ranking** with a transparent score

## Project Structure
```
.
├── data
│   ├── resumes.csv
│   ├── job_description.txt
│   └── skills.json
├── Resume.csv
├── app.py
├── outputs
│   ├── ranking.csv
│   └── report.md
├── notebooks
│   └── Resume_Screening_Demo.ipynb
├── src
│   ├── io_utils.py
│   ├── job_parser.py
│   ├── role_predictor.py
│   ├── report.py
│   ├── screen.py
│   ├── scoring.py
│   ├── skills.py
│   └── text_utils.py
└── requirements.txt
```

## How Scoring Works
The final score is a weighted combination of:
- **Text similarity** between resume and job description (TF-IDF + cosine similarity)
- **Skill match ratio** based on required skills found in the resume

Formula:
```
final_score = 0.6 * similarity_score + 0.4 * skill_match_score
```

You can change the weights via CLI flags.

## Run The System
1. Install dependencies:
```
pip install -r requirements.txt
```
If you prefer a lighter install, you can omit `spacy`, `nltk`, and `PyPDF2` and still run the core pipeline on plain-text resumes.

2. Run the ranking (CSV input):
```
python src/screen.py --resumes Resume.csv --job data/job_description.txt --skills data/skills.json --out outputs/ranking.csv
```

3. Run the ranking (folder of .txt/.pdf resumes):
```
python src/screen.py --resume_folder path/to/resumes --job data/job_description.txt --skills data/skills.json
```
PDF parsing uses `PyPDF2`. If you only use `.txt` resumes, PDF support is optional.

## Web App (Streamlit)
Run the interactive UI:
```
streamlit run app.py
```
This UI lets you paste a job description, upload a resume, and see match score, predicted role, and skill gaps.

## Output
The script creates `outputs/ranking.csv` with:
- `total_score`
- `similarity_score`
- `skill_match_score`
- `matched_required`
- `missing_required`
- `matched_nice`
- `missing_nice`
- `job_terms_overlap`

The script also writes a human-readable report to `outputs/report.md` so you can explain rankings to non-technical stakeholders.

## Notebook
Open `notebooks/Resume_Screening_Demo.ipynb` to explore the pipeline step-by-step in Jupyter.

## Customizing
- Replace `data/resumes.csv` with your own resume data
- Update `data/job_description.txt`
- Extend the skill list in `data/skills.json`
- Adjust scoring weights with:
```
python src/screen.py --weight_similarity 0.7 --weight_skill 0.3
```
 - Control required vs nice-to-have weighting:
```
python src/screen.py --required_weight 0.8
```

## Resume Data Format
Your CSV should have at least one text column. Supported defaults:
- `text` (preferred)
- `resume` or `Resume_str` (auto-detected)

You can also provide a `name` column for candidate names and an `id` column for stable IDs.

For the LiveCareer dataset you described:
- `Resume_str` is detected automatically
- `ID` is mapped to `id`
- `Category` is mapped to `category`

Example (filter to a category):
```
python src/screen.py --resumes Resume.csv --category "Information-Technology"
```

## Notes
This is an educational MVP. In production, you might add:
- richer skill taxonomies
- PDF parsing pipelines
- role-specific weight tuning
- bias and fairness audits
