const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const status = document.getElementById("status");
const result = document.getElementById("result");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const button = form.querySelector("button");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  button.disabled = true;
  status.hidden = false;
  status.classList.remove("error");
  status.textContent = "Thinking...";
  result.hidden = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    answerEl.textContent = data.answer;
    sourcesEl.innerHTML = "";
    for (const s of data.sources) {
      const li = document.createElement("li");
      li.textContent = `${s.source} — p.${s.page} (score ${s.score.toFixed(3)})`;
      sourcesEl.appendChild(li);
    }

    status.hidden = true;
    result.hidden = false;
  } catch (err) {
    status.textContent = err.message || "Something went wrong.";
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
});
