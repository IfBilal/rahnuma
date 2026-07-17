"use client";

import { FormEvent, useEffect, useState } from "react";
import { SignInButton, SignUpButton, UserButton, useAuth } from "@clerk/nextjs";

type Message = { role: "user" | "assistant"; content: string };
type Profile = Record<string, string | number | string[]>;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const THREAD_ID_KEY = "rahnuma-thread-id";

function stableId(key: string) {
  if (typeof window === "undefined") return "";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

export default function Home() {
  const { getToken, isLoaded, userId } = useAuth();
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Assalamualaikum — I’m Rahnuma. Ask about admissions, fees, scholarships, or share your marks for a merit calculation." },
  ]);
  const [question, setQuestion] = useState("");
  const [progress, setProgress] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [profile, setProfile] = useState<Profile>({});
  const [pendingProfile, setPendingProfile] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  // Next renders the initial shell on the server too. localStorage is only
  // available after hydration, so create stable conversation IDs in an effect.
  const [threadId, setThreadId] = useState("");

  useEffect(() => {
    setThreadId(stableId(THREAD_ID_KEY));
  }, []);

  async function refreshProfile() {
    if (!userId) return;
    const token = await getToken();
    const response = await fetch(`${API_BASE}/profiles/me`, { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) throw new Error("Could not load your profile.");
    const data = await response.json();
    setProfile(data.profile);
  }

  useEffect(() => { void refreshProfile().catch(() => undefined); }, [userId]);

  async function sendQuestion(event: FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isLoading || !threadId) return;

    setQuestion("");
    setError("");
    setProgress("Connecting to Rahnuma...");
    setIsLoading(true);
    setMessages((current) => [...current, { role: "user", content: trimmed }, { role: "assistant", content: "" }]);

    try {
      const token = await getToken();
      if (!token) throw new Error("Your session expired. Please sign in again.");
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ question: trimmed, thread_id: threadId }),
      });
      if (!response.ok || !response.body) throw new Error("The advisor could not start. Is the API running?");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";

      const handleEvent = (raw: string) => {
        const line = raw.split("\n").find((item) => item.startsWith("data: "));
        if (!line) return;
        const data = JSON.parse(line.slice(6));
        if (data.type === "progress") setProgress(data.message);
        if (data.type === "token") {
          answer += data.content;
          setMessages((current) => [...current.slice(0, -1), { role: "assistant", content: answer }]);
        }
        if (data.type === "done") {
          // Some worker paths do not emit token events. Keep their final answer.
          if (!answer && data.answer) setMessages((current) => [...current.slice(0, -1), { role: "assistant", content: data.answer }]);
          if (data.profile_confirmation_pending) setPendingProfile(data.profile_confirmation_details);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        events.forEach(handleEvent);
        if (done) break;
      }
      setProgress("");
    } catch (caught) {
      setMessages((current) => current.slice(0, -1));
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  async function confirmProfile(approved: boolean) {
    try {
      const token = await getToken();
      if (!token) throw new Error("Your session expired. Please sign in again.");
      const response = await fetch(`${API_BASE}/chat/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ thread_id: threadId, approved }),
      });
      if (!response.ok) throw new Error("Could not save that profile update.");
      setPendingProfile(null);
      await refreshProfile();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not confirm profile update.");
    }
  }

  if (!isLoaded) return <main className="auth-shell"><p>Loading your admissions desk…</p></main>;

  if (!userId) return <main className="auth-shell"><header className="auth-nav"><div className="brand"><div className="brand-mark">ر</div><div><p className="eyebrow">ADMISSIONS, GROUNDED</p><h1>Rahnuma</h1></div></div><SignInButton mode="modal"><button className="nav-signin">Sign in</button></SignInButton></header><section className="auth-hero"><div className="auth-copy"><p className="eyebrow">THE ADMISSIONS DESK, REBUILT</p><h2>Choose your future<br />with <em>proof.</em></h2><p>Stop comparing hearsay. Rahnuma turns official prospectuses, merit formulas, and your actual profile into a clear next move.</p><div className="auth-actions"><SignUpButton mode="modal"><button>Build my shortlist <span>↗</span></button></SignUpButton><span>No credit card. Your profile stays yours.</span></div></div><div className="auth-proof"><div className="proof-number">05</div><p>seed universities<br />grounded in source docs</p><div className="proof-rule" /><div className="proof-row"><b>FAST</b><span>Merit calculator</span><i>↗</i></div><div className="proof-row"><b>NUST</b><span>Programs & scholarships</span><i>↗</i></div><div className="proof-row"><b>COMSATS</b><span>Fees & admissions</span><i>↗</i></div><small>Every number is either cited or calculated.</small></div></section><footer className="auth-footer"><span>BUILT FOR PAKISTANI STUDENTS</span><span>OFFICIAL SOURCES · HUMAN CONFIRMATION · NO GUESSWORK</span></footer></main>;

  return (
    <main>
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">ر</div><div><p className="eyebrow">ADMISSIONS, GROUNDED</p><h1>Rahnuma</h1></div><div className="user-menu"><UserButton /></div></div>
        <p className="brand-copy">A clearer route through Pakistan&apos;s university admissions.</p>
        <section className="profile"><div className="section-title"><div><p className="eyebrow">REMEMBERS YOU</p><h2>Your profile</h2></div><button onClick={() => void refreshProfile()} aria-label="Refresh profile">↻</button></div>
          {Object.keys(profile).length === 0 ? <p className="muted">Share your marks, city, budget, or university preferences. Rahnuma will ask before saving them.</p> : <dl>{Object.entries(profile).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{Array.isArray(value) ? value.join(", ") : String(value)}</dd></div>)}</dl>}
        </section>
        <section className="trust"><span className="trust-icon">✓</span><p><strong>Evidence first.</strong><br />Every answer is grounded in official prospectuses or labelled live sources.</p></section>
        <p className="source-note">Not an admissions office. Always verify time-sensitive details with the university.</p>
      </aside>
      <section className="chat-shell">
        <header><div><p className="eyebrow">YOUR ADMISSIONS DESK</p><h2>Make your next move<br /><em>with certainty.</em></h2></div><span className="status"><i /> Advisor online</span></header>
        <div className="messages" aria-live="polite">
          {messages.length === 1 && <div className="starter-grid"><button onClick={() => setQuestion("What are FAST's BS Computing admission requirements?")}>Explore a university <b>↗</b></button><button onClick={() => setQuestion("My matric is 90%, FSc is 85%, and test score is 80%. Calculate my FAST merit.")}>Calculate my merit <b>↗</b></button><button onClick={() => setQuestion("What scholarships are available at NUST?")}>Find scholarships <b>↗</b></button></div>}
          {messages.map((message, index) => <article className={`message ${message.role}`} key={index}>{message.role === "assistant" && <span className="message-mark">ر</span>}<div>{message.content || <span className="typing">Working<span>.</span><span>.</span><span>.</span></span>}</div></article>)}
          {progress && <p className="progress">{progress}</p>}
          {error && <p className="error">{error}</p>}
        </div>
        {pendingProfile && <section className="confirm"><div><strong>Save this profile update?</strong><pre>{JSON.stringify(pendingProfile.proposed_changes, null, 2)}</pre></div><div><button className="ghost" onClick={() => void confirmProfile(false)}>Not now</button><button onClick={() => void confirmProfile(true)}>Save profile</button></div></section>}
        <form onSubmit={sendQuestion}><div className="input-wrap"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder="Ask about admissions, merit, fees, or scholarships…" rows={2} /><span><b>Enter</b> to send · <b>Shift + Enter</b> for a new line</span></div><button type="submit" disabled={isLoading || !question.trim() || !threadId} aria-label="Ask Rahnuma"><span>{isLoading ? "···" : "↑"}</span></button></form>
        <p className="composer-note">Rahnuma cites the source behind its claims. Your profile is never saved without confirmation.</p>
      </section>
    </main>
  );
}
