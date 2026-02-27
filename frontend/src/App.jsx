import { useState } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!question.trim()) return

    setLoading(true)
    setResponse('')

    try {
      const res = await fetch('http://localhost:8000/ask/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      })

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()

        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        setResponse((prev) => prev + chunk)
        setLoading(false) // Turn off loading once first chunk arrives
      }
    } catch (error) {
      console.error('Error:', error)
      setResponse(`Error: ${error.message}`)
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>FastAPI Q&A</h1>
      <p className="subtitle">Ask questions about FastAPI documentation</p>

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Search..."
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </form>

      {loading && !response && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Thinking...</p>
        </div>
      )}

      {response && (
        <div className="response">
          <h3>Answer:</h3>
          <div className="answer">{response}</div>
        </div>
      )}
    </div>
  )
}

export default App
