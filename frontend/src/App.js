import { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");

  const ask = async () => {
    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({question})
    });
    const data = await res.json();
    setResponse(data.answer);
  };

  return (
    <div style={{padding:20}}>
      <h2>HyDE RAG</h2>
      <input value={question} onChange={e=>setQuestion(e.target.value)} />
      <button onClick={ask}>Ask</button>
      <p>{response}</p>
    </div>
  );
}

export default App;
