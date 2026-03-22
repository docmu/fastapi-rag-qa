# FastAPI RAG Q&A Frontend

React frontend with real-time streaming responses using Fetch API with ReadableStream.

## Features

- **Real-time streaming** - Displays AI responses as they arrive, token by token
- **Clean UI** - Minimal, readable design
- **Loading states** - Shows spinner while waiting for first response chunk
- **Error handling** - Gracefully handles network and API errors

## Technologies Used

- **React** - UI library
- **Vite** - Build tool and dev server
- **ReadableStream API** - Streaming responses

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

## How Streaming Works

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

## Extending This Frontend

### Ideas for Enhancement

**Stop Generation:**
   ```javascript
   const abortController = new AbortController()
   fetch(url, { signal: abortController.signal })
   // Add "Stop" button that calls abortController.abort()
   ```

**Show Sources:**
   ```javascript
   // Switch to /ask endpoint, display returned sources
   // Link to original documentation
   ```

**Markdown Rendering:**
   ```bash
   npm install react-markdown
   ```
   ```jsx
   import ReactMarkdown from 'react-markdown'
   <ReactMarkdown>{response}</ReactMarkdown>
   ```

**Copy to Clipboard:**
   ```javascript
   const copyResponse = () => {
     navigator.clipboard.writeText(response)
   }
   ```

## Learn More
- [Fetch API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [Streams API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API)
- [Vite Documentation](https://vitejs.dev/)
