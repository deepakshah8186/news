
# 💬 Financial News Chat Application

This is a simple GenAI-powered chat application that allows users to ask questions about recent financial news and receive relevant, summarized responses.

## 📌 Project Description

The application ingests financial news data from a structured JSON file and leverages a language model (via LangChain and FAISS) to answer user queries based on this content.

- Users type questions in a simple web UI.
- The app semantically searches news articles for relevant context.
- It responds with a summarized answer, grounded in the retrieved content.

## 🚀 Features

- Chat interface built using **Streamlit**
- Financial news ingestion from JSON file
- Embedding and retrieval using **LangChain + FAISS**
- Answer generation using **OpenAI LLMs**
- Lightweight UI optimized for functionality over styling

## 📂 Input

Place a JSON file named `stock_news.json` in the root directory. The file should be structured like this:

```json
{
  "AAPL": [
    {
      "title": "Apple releases new iPhone", 
      "ticker": "AAPL",
      "full_text": "Apple Inc. today announced the launch of its latest iPhone...",
      "link": "https://example.com/apple-news"
    }
  ]

}
```

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI (via API Key)
- FAISS (Vector Store)

## 🧪 Running the App

1. Clone the repository:
   ```bash
   git clone https://github.com/deepakshah8186/news.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenAI API key to your environment:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

4. Run the app:
   ```bash
   streamlit run stock-news-app.py
   ```

## 📬 License

This project is for evaluation purposes only.
