import { useState } from "react";
import "./App.css";

function App() {
  const [title, setTitle] = useState("");
  const [article, setArticle] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const checkNews = async () => {
    if (!article.trim()) {
      alert("Please enter a news article.");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: title,
          article: article,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (error) {
      alert("Backend connection failed. Make sure Flask is running.");
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setTitle("");
    setArticle("");
    setResult(null);
  };

  return (
    <div className="app">

      <nav className="navbar">
        <div className="logo">📰 TruthCheck</div>

        <div className="nav-links">
          <span>Home</span>
          <span>How It Works</span>
          <span>About</span>
        </div>
      </nav>

      <main className="container">

        <section className="hero">
          <p className="badge">AI POWERED NEWS ANALYZER</p>

          <h1>Fake News Detection</h1>

          <p>
            Check whether a news article is likely to be
            <b> real or fake </b>
            using Machine Learning.
          </p>
        </section>

        <section className="card">

          <label>News Headline</label>

          <input
            type="text"
            placeholder="Enter the news headline..."
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />

          <label>News Article</label>

          <textarea
            placeholder="Paste the complete news article here..."
            value={article}
            onChange={(e) => setArticle(e.target.value)}
          />

          <div className="buttons">

            <button
              className="check-btn"
              onClick={checkNews}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "🔍 Check News"}
            </button>

            <button className="clear-btn" onClick={clearAll}>
              Clear
            </button>

          </div>

          {result && (
            <div
              className={
                result.result === "REAL"
                  ? "result real"
                  : "result fake"
              }
            >
              <div className="result-icon">
                {result.result === "REAL" ? "✓" : "!"}
              </div>

              <div>
                <h2>
                  {result.result === "REAL"
                    ? "Likely Real News"
                    : "Likely Fake News"}
                </h2>

                <p>
                  Model confidence:
                  <strong> {result.confidence}%</strong>
                </p>
              </div>
            </div>
          )}

        </section>

        <section className="features">

          <div>
            <h3>🤖 Machine Learning</h3>
            <p>Uses TF-IDF and Logistic Regression.</p>
          </div>

          <div>
            <h3>⚡ Fast Analysis</h3>
            <p>Get a prediction within seconds.</p>
          </div>

          <div>
            <h3>🔒 Simple & Easy</h3>
            <p>Paste an article and check the result.</p>
          </div>

        </section>

      </main>

      <footer>
        Fake News Detection • Machine Learning Project
      </footer>

    </div>
  );
}

export default App;