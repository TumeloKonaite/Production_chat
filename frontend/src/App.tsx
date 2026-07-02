import { useState } from "react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

type TavusConversationResponse = {
  conversation_id: string;
  conversation_url: string;
};

export default function App() {
  const [visitorName, setVisitorName] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversationUrl, setConversationUrl] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function startConversation() {
    setIsStarting(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/tavus/conversations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          visitor_name: visitorName || "Website visitor",
        }),
      });

      if (!response.ok) {
        const message = await readErrorMessage(response);
        throw new Error(message);
      }

      const data = (await response.json()) as TavusConversationResponse;
      setConversationId(data.conversation_id);
      setConversationUrl(data.conversation_url);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Unable to start the video conversation.",
      );
    } finally {
      setIsStarting(false);
    }
  }

  async function endConversation() {
    if (!conversationId) {
      return;
    }

    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/tavus/conversations/end`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        const message = await readErrorMessage(response);
        throw new Error(message);
      }

      setConversationId(null);
      setConversationUrl(null);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Unable to end the video conversation.",
      );
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Local Tavus test client</p>
          <h1>Talk to my AI video avatar</h1>
          <p className="subtitle">
            Ask about my projects, experience, skills, education, and AI engineering
            work.
          </p>
        </div>

        <div className="control-card">
          <label className="field">
            <span>Your name</span>
            <input
              type="text"
              value={visitorName}
              onChange={(event) => setVisitorName(event.target.value)}
              placeholder="Optional"
              autoComplete="name"
            />
          </label>

          <div className="actions">
            <button
              type="button"
              className="primary-button"
              onClick={startConversation}
              disabled={isStarting || Boolean(conversationUrl)}
            >
              {isStarting ? "Starting..." : "Start video conversation"}
            </button>

            <button
              type="button"
              className="secondary-button"
              onClick={endConversation}
              disabled={!conversationId}
            >
              End conversation
            </button>
          </div>

          <div className="status-area">
            <p className="backend-note">Backend: {BACKEND_URL}</p>
            {conversationId ? (
              <p className="conversation-pill">Conversation: {conversationId}</p>
            ) : null}
            {error ? <p className="error-banner">{error}</p> : null}
          </div>
        </div>

        <div className="frame-panel">
          {conversationUrl ? (
            <iframe
              src={conversationUrl}
              title="Tavus AI Avatar Conversation"
              allow="camera; microphone; fullscreen; display-capture"
              className="conversation-frame"
            />
          ) : (
            <div className="frame-placeholder">
              <p>The Tavus conversation will appear here after you start it.</p>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {
    // Ignore JSON parsing failures and fall back to a generic message.
  }

  return "Request failed. Check that the backend is running and Tavus is configured.";
}
