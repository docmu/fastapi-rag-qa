# FastAPI RAG Q&A Frontend

A minimal React frontend for the FastAPI RAG Q&A system, demonstrating real-time streaming responses using the Fetch API with ReadableStream.

## Features

- **Real-time streaming** - Displays AI responses as they arrive, token by token
- **Clean UI** - Minimal, readable design
- **Loading states** - Shows spinner while waiting for first response chunk
- **Error handling** - Gracefully handles network and API errors

## How Streaming Works

This frontend demonstrates how to consume Server-Sent Events (SSE) / streaming responses from an API using the modern Fetch API:

```javascript
// 1. Make request with fetch
const res = await fetch('http://localhost:8000/ask/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question }),
})

// 2. Get ReadableStream reader from response body
const reader = res.body.getReader()
const decoder = new TextDecoder()

// 3. Read chunks in a loop
while (true) {
  const { done, value } = await reader.read()
  if (done) break

  // 4. Decode and display each chunk immediately
  const chunk = decoder.decode(value, { stream: true })
  setResponse(prev => prev + chunk)  // Update UI in real-time
}
```

### Key Concepts

**ReadableStream API:**
- `res.body.getReader()` - Gets a reader for the response stream
- `reader.read()` - Returns next chunk as `{ done, value }`
- `TextDecoder` - Converts byte arrays to strings

**Streaming Benefits:**
- Lower perceived latency (users see results immediately)
- Better UX for long responses
- Incremental rendering

## Running the Frontend

### Development

```bash
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

### Prerequisites

Make sure the FastAPI backend is running:

```bash
# In project root
python3 main.py
```

Backend must be at `http://localhost:8000` with CORS enabled.

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx       # Main component with streaming logic
│   ├── App.css       # Styling
│   ├── main.jsx      # Entry point
│   └── index.css     # Global styles
├── package.json
└── vite.config.js
```

## Code Walkthrough

### State Management

```javascript
const [question, setQuestion] = useState('')    // User input
const [response, setResponse] = useState('')    // Streamed response
const [loading, setLoading] = useState(false)   // Loading state
```

### Streaming Handler

```javascript
const handleSubmit = async (e) => {
  e.preventDefault()

  setLoading(true)
  setResponse('')  // Clear previous response

  // Fetch with streaming
  const res = await fetch('http://localhost:8000/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  // Read stream chunk by chunk
  const reader = res.body.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value, { stream: true })
    setResponse(prev => prev + chunk)
    setLoading(false)  // Turn off loading on first chunk
  }
}
```

### UI Components

**Loading State:**
```jsx
{loading && !response && (
  <div className="loading">
    <div className="spinner"></div>
    <p>Thinking...</p>
  </div>
)}
```

**Streaming Response:**
```jsx
{response && (
  <div className="response">
    <h3>Answer:</h3>
    <div className="answer">{response}</div>
  </div>
)}
```

## Extending This Frontend

### Ideas for Enhancement

1. **Conversation History:**
   ```javascript
   const [messages, setMessages] = useState([])
   // Store Q&A pairs, enable multi-turn conversations
   ```

2. **Stop Generation:**
   ```javascript
   const abortController = new AbortController()
   fetch(url, { signal: abortController.signal })
   // Add "Stop" button that calls abortController.abort()
   ```

3. **Show Sources:**
   ```javascript
   // Switch to /ask endpoint, display returned sources
   // Link to original documentation
   ```

4. **Markdown Rendering:**
   ```bash
   npm install react-markdown
   ```
   ```jsx
   import ReactMarkdown from 'react-markdown'
   <ReactMarkdown>{response}</ReactMarkdown>
   ```

5. **Copy to Clipboard:**
   ```javascript
   const copyResponse = () => {
     navigator.clipboard.writeText(response)
   }
   ```

## Troubleshooting

### CORS Errors

If you see CORS errors in the console:

1. Make sure FastAPI backend has CORS middleware enabled
2. Check that `allow_origins` includes `http://localhost:5173`
3. Verify backend is running on port 8000

### Streaming Not Working

If the response appears all at once instead of streaming:

1. Check that you're calling `/ask/stream` not `/ask`
2. Verify FastAPI is using `StreamingResponse`
3. Check browser console for errors

### Connection Refused

If fetch fails with "Connection refused":

1. Make sure backend is running: `python3 main.py`
2. Check backend is on port 8000: `http://localhost:8000`
3. Try the health check: `http://localhost:8000/`

## Technologies Used

- **React** - UI library
- **Vite** - Build tool and dev server
- **Fetch API** - HTTP requests
- **ReadableStream API** - Streaming responses

## Learn More

- [Fetch API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [Streams API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API)
- [React Hooks](https://react.dev/reference/react)
- [Vite Documentation](https://vitejs.dev/)
