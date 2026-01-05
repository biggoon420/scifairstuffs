function qp() {
  const u = new URL(window.location.href);
  return {
    assignmentId: u.searchParams.get("assignmentId"),
    workerId: u.searchParams.get("workerId"),
    hitId: u.searchParams.get("hitId"),
    turkSubmitTo: u.searchParams.get("turkSubmitTo")
  };
}

function setDisabled(disabled) {
  ["playA","playB","skipA","skipB","voteA","voteB"].forEach(id => {
    const el = document.getElementById(id);
    el.disabled = disabled;
    el.style.opacity = disabled ? "0.55" : "1.0";
    el.style.cursor = disabled ? "not-allowed" : "pointer";
  });
}

async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  const j = await res.json();
  if (!res.ok) throw new Error(j.detail || "Request failed");
  return j;
}

let SESSION = null;
let PARAMS = qp();

function stopAll() {
  const a = document.getElementById("audioA");
  const b = document.getElementById("audioB");
  a.pause();
  b.pause();
  a.currentTime = 0;
  b.currentTime = 0;
}

function renderPair(state) {
  const status = document.getElementById("status");
  const pairInfo = document.getElementById("pairInfo");
  const progress = document.getElementById("progress");

  if (state.status === "preview") {
    status.textContent = state.message;
    pairInfo.textContent = "";
    progress.textContent = "";
    setDisabled(true);
    return;
  }

  if (state.status === "finished") {
    setDisabled(true);
    document.getElementById("finish").classList.remove("hidden");
    document.getElementById("code").textContent = state.completion_code;
    document.getElementById("completionCodeField").value = state.completion_code;

    const aId = PARAMS.assignmentId || "";
    document.getElementById("assignmentIdField").value = aId;

    const submitForm = document.getElementById("submitForm");
    const noSubmit = document.getElementById("noSubmit");

    if (PARAMS.turkSubmitTo && aId) {
      submitForm.action = PARAMS.turkSubmitTo.replace(/\/+$/, "") + "/mturk/externalSubmit";
      noSubmit.classList.add("hidden");
    } else {
      submitForm.classList.add("hidden");
      noSubmit.classList.remove("hidden");
      noSubmit.textContent = "Local test mode: copy the completion code above and submit it manually in MTurk.";
    }

    status.textContent = "All comparisons complete.";
    pairInfo.textContent = "";
    progress.textContent = `${state.done_n}/${state.target_n}`;
    return;
  }

  status.textContent = `Session ${state.done_n}/${state.target_n}`;
  progress.textContent = `Progress: ${state.done_n}/${state.target_n}`;

  const p = state.pair;
  pairInfo.textContent = `A=${p.a_file} (${p.a_chunk_idx}/${p.a_chunk_total})  vs  B=${p.b_file} (${p.b_chunk_idx}/${p.b_chunk_total})`;

  const audioA = document.getElementById("audioA");
  const audioB = document.getElementById("audioB");
  audioA.src = p.a_url;
  audioB.src = p.b_url;

  setDisabled(false);
}

async function start() {
  setDisabled(true);
  const state = await postJSON("/api/start", PARAMS);
  SESSION = state.session_id;
  renderPair(state);
}

async function vote(winner) {
  if (!SESSION) return;
  stopAll();
  const state = await postJSON("/api/vote", { session_id: SESSION, winner });
  renderPair(state);
}

async function skip(which) {
  if (!SESSION) return;
  stopAll();
  const state = await postJSON("/api/skip", { session_id: SESSION, which });
  renderPair(state);
}

window.addEventListener("load", () => {
  document.getElementById("playA").addEventListener("click", () => {
    stopAll();
    document.getElementById("audioA").play();
  });
  document.getElementById("playB").addEventListener("click", () => {
    stopAll();
    document.getElementById("audioB").play();
  });
  document.getElementById("voteA").addEventListener("click", () => vote(1));
  document.getElementById("voteB").addEventListener("click", () => vote(2));
  document.getElementById("skipA").addEventListener("click", () => skip("A"));
  document.getElementById("skipB").addEventListener("click", () => skip("B"));

  start().catch(e => {
    document.getElementById("status").textContent = "Error: " + String(e.message || e);
    setDisabled(true);
  });
});

