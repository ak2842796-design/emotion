(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const input = $("inputText");
  const analyzeBtn = $("analyzeBtn");
  const analyzeLabel = $("analyzeLabel");
  const clearBtn = $("clearBtn");
  const charCount = $("charCount");
  const emptyState = $("emptyState");
  const resultContent = $("resultContent");
  const loadingState = $("loadingState");
  const errorState = $("errorState");
  const resultBadge = $("resultBadge");
  const probabilityList = $("probabilityList");
  const colors = {
    joy: "#19b780", love: "#df5aa7", surprise: "#8765e8",
    sadness: "#4e8bea", fear: "#e9aa37", anger: "#e45d70"
  };
  const symbols = {joy:"☺",love:"♡",surprise:"✧",sadness:"☂",fear:"◌",anger:"⚡"};
  const labels = ["anger", "fear", "joy", "love", "sadness", "surprise"];
  let toastTimer;

  function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
  }
  function setView(view) {
    emptyState.classList.toggle("hidden", view !== "empty");
    resultContent.classList.toggle("hidden", view !== "result");
    loadingState.classList.toggle("hidden", view !== "loading");
    errorState.classList.toggle("hidden", view !== "error");
  }
  function setStatus(online, detail = "") {
    $("sideStatusOrb").style.background = online ? "#1ab486" : "#df5b6b";
    $("sideStatusOrb").style.boxShadow = online ? "0 0 0 4px #1ab48618" : "0 0 0 4px #df5b6b18";
    $("sideStatusText").textContent = online ? "API connected" : "API unavailable";
    $("topStatus").textContent = online ? "API online" : "API offline";
    document.querySelector(".live-dot").style.background = online ? "#19b887" : "#df5b6b";
    $("healthPill").className = "status-pill " + (online ? "online" : "offline");
    $("healthPillText").textContent = online ? "Operational" : "Offline";
    $("serviceIndicator").className = "service-indicator " + (online ? "online" : "offline");
    $("healthDetail").textContent = detail || (online ? "Model and tokenizer are loaded." : "Unable to reach /health.");
    $("lastChecked").textContent = "Checked " + new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
  }
  async function checkHealth(silent = true) {
    try {
      const response = await fetch("/health", {headers: {"Accept":"application/json"}});
      if (!response.ok) throw new Error("Health endpoint returned HTTP " + response.status);
      const data = await response.json();
      const online = data.status === "ok" && data.model_loaded === true;
      setStatus(online, online ? "FastAPI is responding; model and tokenizer are loaded." : "API responds, but the model may not be loaded.");
      if (!online && !silent) showToast("API is responding, but the model is not ready yet.");
      return online;
    } catch (err) {
      setStatus(false, "Start the server with: uvicorn main:app --reload");
      if (!silent) showToast("Could not connect to the API. Is the server running?");
      return false;
    }
  }
  function renderProbabilities(probabilities) {
    probabilityList.innerHTML = "";
    const ordered = labels
      .filter(label => Object.prototype.hasOwnProperty.call(probabilities, label))
      .map(label => [label, Number(probabilities[label])])
      .sort((a,b) => b[1] - a[1]);
    ordered.forEach(([label, score], index) => {
      const safeScore = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
      const row = document.createElement("div");
      row.className = "prob-row";
      const name = document.createElement("div");
      name.className = "prob-name";
      const dot = document.createElement("span");
      dot.className = "prob-dot";
      dot.style.background = colors[label] || "#8390a8";
      const labelText = document.createElement("span");
      labelText.textContent = label;
      name.append(dot, labelText);
      const track = document.createElement("div");
      track.className = "prob-track";
      const fill = document.createElement("div");
      fill.className = "prob-fill";
      fill.style.background = colors[label] || "#8390a8";
      fill.style.width = "0%";
      track.append(fill);
      const value = document.createElement("div");
      value.className = "prob-value";
      value.textContent = (safeScore * 100).toFixed(2) + "%";
      row.append(name, track, value);
      probabilityList.append(row);
      requestAnimationFrame(() => { fill.style.width = (safeScore * 100) + "%"; });
    });
  }
  async function analyze() {
    const text = input.value.trim();
    if (!text) {
      input.focus();
      showToast("Enter a sentence first, then analyze it.");
      return;
    }
    if (text.length > 2000) {
      showToast("Please keep your text within 2,000 characters.");
      return;
    }
    analyzeBtn.disabled = true;
    analyzeLabel.textContent = "Analyzing…";
    setView("loading");
    resultBadge.className = "result-status";
    resultBadge.innerHTML = "<span></span> PROCESSING";
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json", "Accept":"application/json"},
        body: JSON.stringify({text})
      });
      let data;
      try { data = await response.json(); } catch { throw new Error("The API returned an unreadable response."); }
      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "Request failed (HTTP " + response.status + ").";
        throw new Error(detail);
      }
      if (!data.predicted_emotion || typeof data.confidence !== "number" || !data.all_probabilities) {
        throw new Error("The response format did not match the expected prediction schema.");
      }
      const emotion = String(data.predicted_emotion).toLowerCase();
      $("predictedEmotion").textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
      $("predictionSubtext").textContent = "Highest-scoring class from the BiGRU model";
      $("emotionSymbol").textContent = symbols[emotion] || "✦";
      $("emotionSymbol").style.color = colors[emotion] || "#6475e8";
      $("emotionSymbol").style.background = (colors[emotion] || "#6475e8") + "20";
      $("predictedEmotion").style.color = colors[emotion] || "#6475e8";
      const confidence = Math.max(0, Math.min(1, data.confidence));
      $("confidenceValue").textContent = (confidence * 100).toFixed(2) + "%";
      $("confidenceBar").style.width = (confidence * 100) + "%";
      renderProbabilities(data.all_probabilities);
      resultBadge.className = "result-status success";
      resultBadge.innerHTML = "<span></span> ANALYSIS COMPLETE";
      setView("result");
      checkHealth(true);
    } catch (err) {
      $("errorMessage").textContent = err.message || "Unexpected error. Please try again.";
      resultBadge.className = "result-status";
      resultBadge.innerHTML = "<span></span> REQUEST FAILED";
      setView("error");
    } finally {
      analyzeBtn.disabled = false;
      analyzeLabel.textContent = "Analyze emotion";
    }
  }
  input.addEventListener("input", () => { charCount.textContent = input.value.length; });
  analyzeBtn.addEventListener("click", analyze);
  input.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
  });
  clearBtn.addEventListener("click", () => {
    input.value = "";
    charCount.textContent = "0";
    setView("empty");
    resultBadge.className = "result-status";
    resultBadge.innerHTML = "<span></span> WAITING";
    input.focus();
  });
  document.querySelectorAll(".example").forEach(button => {
    button.addEventListener("click", () => {
      input.value = button.dataset.text || "";
      charCount.textContent = input.value.length;
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    });
  });
  $("retryBtn").addEventListener("click", analyze);
  $("refreshHealth").addEventListener("click", () => checkHealth(false));
  checkHealth(true);
  setInterval(() => checkHealth(true), 30000);
})();