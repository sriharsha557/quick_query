Absolutely 👍 — adding your **Quick Query logo** or a screenshot at the top of the README will make the repo look much more polished.

Here’s the updated **README.md** with a placeholder for your logo:

---

````markdown
# 🔍 Quick Query

<p align="center">
  <img src="assets/quickquery_logo.png" alt="Quick Query Logo" width="200"/>
</p>

**Quick Query** is an interactive document search assistant built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), and [CrewAI](https://github.com/joaomdmoura/crewai).  
It allows you to upload documents (PDF, DOCX, etc.) and ask natural language questions to quickly find answers inside them.

---

## 🚀 Features
- Upload and process documents (PDF, Word).
- Automatic text splitting and embedding with HuggingFace.
- Vector search powered by **ChromaDB**.
- Natural language Q&A using **Groq LLM** through LangChain.
- Clean and simple UI built with **Streamlit**.
- Predefined topic list for easier navigation.

---

## 🛠️ Tech Stack
- **Frontend/UI:** Streamlit  
- **Orchestration:** CrewAI  
- **LLM:** Groq (via `langchain-groq`)  
- **Embeddings:** HuggingFace + Sentence Transformers  
- **Vector Store:** ChromaDB  
- **Document Processing:** PyPDF2, python-docx  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/quick-query.git
cd quick-query
````

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Setup

Create a `.env` file in the project root with your API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the App

Start the Streamlit server:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deployment

The app is configured to run on **Streamlit Cloud**.
Just push your code along with `requirements.txt` to GitHub, and connect the repo to Streamlit Cloud.

---

## 📚 Example Use Cases

* Search and summarize research papers.
* Extract insights from business documents.
* Quickly find relevant sections in PDFs or Word files.
* Knowledge base Q\&A for teams.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request if you’d like to improve the project.

---

## 📜 License

This project is licensed under the MIT License.

````


### 🔹 How to use your logo
1. Place your logo image (e.g., `quickquery_logo.png`) inside an `assets/` folder in your repo.  
2. Update the path in the README:  
   ```markdown
   <img src="assets/quickquery_logo.png" alt="Quick Query Logo" width="200"/>
````

3. Commit and push to GitHub.

---
