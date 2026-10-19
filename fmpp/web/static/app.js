const leagueFilter = document.querySelector("#league-filter");
const matchList = document.querySelector("#match-list");
const scheduleMessage = document.querySelector("#schedule-message");
const predictionDialog = document.querySelector("#prediction-dialog");
const predictionDetail = document.querySelector("#prediction-detail");
const dayTabs = [...document.querySelectorAll(".tab")];
let currentDay = "today";
let allTeams = [];
let visibleTeamCount = window.matchMedia("(max-width: 650px)").matches ? 10 : 40;

async function api(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Permintaan gagal.");
  return data;
}

async function loadCompetitions() {
  try {
    const competitions = await api("/api/competitions");
    for (const item of competitions) {
      const option = document.createElement("option");
      option.value = item.code;
      option.textContent = item.name;
      leagueFilter.append(option);
    }
  } catch (error) {
    scheduleMessage.textContent = error.message;
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

async function loadMatches() {
  matchList.replaceChildren();
  scheduleMessage.textContent = "Memuat jadwal lokal…";
  const query = new URLSearchParams({ day: currentDay });
  if (leagueFilter.value) query.set("league", leagueFilter.value);
  try {
    const fixtures = await api(`/api/matches?${query}`);
    scheduleMessage.textContent = fixtures.length ? "" : "Jadwal tidak ditemukan. Perbarui data melalui CLI.";
    for (const fixture of fixtures) {
      const card = document.createElement("article");
      card.className = "match-card";
      card.innerHTML = `
        <div class="match-meta"><span>${escapeHtml(fixture.competition)}</span>
          <time>${escapeHtml(fixture.date_wib.slice(11, 16) || "Waktu belum tersedia")}</time></div>
        <div class="teams"><span>${escapeHtml(fixture.home_team)}</span><span class="vs">VS</span>
          <span>${escapeHtml(fixture.away_team)}</span></div>
        <div class="card-actions"><span class="confidence">Analisis probabilitas</span>
          <button class="detail-button" data-id="${fixture.fixture_id}">Lihat analisis →</button></div>`;
      card.querySelector("button").addEventListener("click", () => showPrediction(fixture.fixture_id));
      matchList.append(card);
    }
  } catch (error) {
    scheduleMessage.textContent = error.message;
  }
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }

function matrixHtml(matrix) {
  const max = Math.max(...matrix.flat());
  return `<div class="matrix" aria-label="Heatmap probabilitas skor">${matrix.map((row, home) =>
    row.map((value, away) => {
      const alpha = max ? 0.12 + value / max * 0.78 : 0.12;
      return `<span class="matrix-cell" title="${home}-${away}: ${percent(value)}"
        style="background:rgba(187,243,107,${alpha})">${percent(value)}</span>`;
    }).join("")).join("")}</div><p>Baris: gol kandang · Kolom: gol tandang</p>`;
}

function formHtml(team, matches, analysis) {
  const count = analysis.matches_analyzed;
  const metric = (value) => value == null ? "—" : value.toFixed(2);
  const summary = count
    ? `<p>Form terbaru: <strong>${escapeHtml(analysis.recent_results)}</strong> ·
        M/S/K ${analysis.wins}/${analysis.draws}/${analysis.losses} ·
        poin/laga ${metric(analysis.points_per_game)} ·
        gol cetak/kebobolan ${metric(analysis.avg_goals_scored)}/${metric(analysis.avg_goals_conceded)}.</p>`
    : "";
  const history = matches.length
    ? `<div class="form-list">${matches.map((match) => {
        const teamIsHome = match.team_is_home;
        const opponent = teamIsHome ? match.away_team : match.home_team;
        const teamGoals = teamIsHome ? match.home_score : match.away_score;
        const opponentGoals = teamIsHome ? match.away_score : match.home_score;
        const result = teamGoals > opponentGoals ? "M" : teamGoals === opponentGoals ? "S" : "K";
        return `<div class="form-match"><span class="form-result form-${result.toLowerCase()}">${result}</span>
          <span>${escapeHtml(match.date.slice(0, 10))} · ${teamIsHome ? "Kandang" : "Tandang"} vs
          ${escapeHtml(opponent)} · ${teamGoals}–${opponentGoals}</span></div>`;
      }).join("")}</div>`
    : `<p>Data riwayat ${escapeHtml(team)} belum ditemukan. Form lima laga belum dapat dihitung.</p>`;
  const warning = analysis.minimum_met
    ? ""
    : `<p class="form-warning">Baru tersedia ${count} dari minimal 5 laga sebelumnya.</p>`;
  return `<section class="detail-panel form-panel"><h3>Form ${matches.length} laga terakhir · ${escapeHtml(team)}</h3>
    ${summary}${warning}${history}</section>`;
}

function renderTeamDirectory() {
  const query = document.querySelector("#team-search").value.trim().toLocaleLowerCase();
  const matchingTeams = allTeams.filter((item) =>
    item.team.toLocaleLowerCase().includes(query)
  );
  const visibleTeams = matchingTeams.slice(0, visibleTeamCount);
  document.querySelector("#team-list").innerHTML = visibleTeams.map((item) =>
    formHtml(item.team, item.recent_form, item.team_analysis)
  ).join("");
  document.querySelector("#team-directory-message").textContent = matchingTeams.length
    ? `${matchingTeams.length} tim ditemukan · menampilkan ${visibleTeams.length}.`
    : "Tidak ada tim yang cocok dengan pencarian.";
  document.querySelector("#load-more-teams").hidden =
    visibleTeams.length >= matchingTeams.length;
}

async function loadTeamDirectory() {
  const message = document.querySelector("#team-directory-message");
  message.textContent = "Memuat riwayat semua tim…";
  try {
    const result = await api("/api/teams?last_n=5");
    allTeams = result.teams;
    renderTeamDirectory();
  } catch (error) {
    message.textContent = error.message;
  }
}

document.querySelector("#team-search").addEventListener("input", () => {
  visibleTeamCount = 40;
  renderTeamDirectory();
});
document.querySelector("#load-more-teams").addEventListener("click", () => {
  visibleTeamCount += window.matchMedia("(max-width: 650px)").matches ? 10 : 40;
  renderTeamDirectory();
});

window.addEventListener("resize", () => {
  visibleTeamCount = window.matchMedia("(max-width: 650px)").matches ? 10 : 40;
  renderTeamDirectory();
});

async function showPrediction(fixtureId) {
  predictionDetail.innerHTML = "<p>Menyiapkan prediksi…</p>";
  predictionDialog.showModal();
  try {
    const lastN = document.querySelector("#evaluation-n").value;
    const data = await api(`/api/matches/${fixtureId}/prediction?last_n=${lastN}`);
    const { fixture, prediction, recent_form: form, team_analysis: analysis } = data;
    const lines = prediction.over_under;
    predictionDetail.innerHTML = `
      <p class="eyebrow">${escapeHtml(fixture.competition)} · ${escapeHtml(fixture.date_wib)}</p>
      <h2 id="detail-title" class="detail-title">${escapeHtml(fixture.home_team)} vs ${escapeHtml(fixture.away_team)}</h2>
      <p class="detail-subtitle">${escapeHtml(data.disclaimer)}</p>
      <div class="detail-grid form-grid">
        ${formHtml(fixture.home_team, form.home, analysis.home)}
        ${formHtml(fixture.away_team, form.away, analysis.away)}
      </div>
      <div class="probability-bars">
        ${[["Menang kandang", prediction.prob_home], ["Seri", prediction.prob_draw], ["Menang tandang", prediction.prob_away]].map(([label, value]) =>
          `<div><div class="bar-label"><span>${label}</span><strong>${percent(value)}</strong></div><div class="bar-track"><div class="bar-fill" style="width:${value * 100}%"></div></div></div>`
        ).join("")}
      </div>
      <div class="detail-grid">
        <section class="detail-panel"><h3>Ekspektasi gol & Over/Under</h3>
          <p>${escapeHtml(fixture.home_team)} ${prediction.lambda_home.toFixed(2)} · ${escapeHtml(fixture.away_team)} ${prediction.lambda_away.toFixed(2)} · Total ${prediction.expected_total.toFixed(2)}</p>
          <p>${["1_5", "2_5", "3_5", "4_5"].map((key) => `O${key.replace("_", ".")}: ${percent(lines[`over_${key}`])}`).join(" · ")}</p>
          <p>Garis seimbang ${prediction.balanced_line.line}: Over ${percent(prediction.balanced_line.over_prob)}</p>
          <p>BTTS Ya ${percent(prediction.prob_btts_yes)} · Tidak ${percent(prediction.prob_btts_no)}</p></section>
        <section class="detail-panel"><h3>Skor teratas & kualitas data</h3>
          <p>${prediction.top_scores.map((score) => `${score.score} (${percent(score.prob)})`).join(" · ")}</p>
          <p>Keyakinan data: ${escapeHtml(prediction.data_confidence)}; riwayat ${prediction.home_matches_played} / ${prediction.away_matches_played} laga.</p>
          ${prediction.data_warning ? `<p>${escapeHtml(prediction.data_warning)}</p>` : ""}</section>
        <section class="detail-panel matrix-panel"><h3>Matriks probabilitas skor</h3>${matrixHtml(prediction.score_matrix)}</section>
      </div>`;
  } catch (error) {
    predictionDetail.innerHTML = `<p class="message">${escapeHtml(error.message)}</p>`;
  }
}

function metricTable(summary) {
  const rows = Object.entries(summary).filter(([, value]) => value && typeof value === "object" && !Array.isArray(value));
  return `<div class="performance-panel"><table><thead><tr><th>Model</th><th>Log Loss 1X2</th><th>Brier 1X2</th><th>RPS</th><th>Akurasi</th><th>MAE gol kandang</th></tr></thead><tbody>
    ${rows.map(([name, metrics]) => `<tr><td>${escapeHtml(name)}</td><td>${metrics.log_loss_1x2?.toFixed(3) ?? "—"}</td>
      <td>${metrics.brier_1x2?.toFixed(3) ?? "—"}</td><td>${metrics.rps_1x2?.toFixed(3) ?? "—"}</td>
      <td>${metrics.accuracy_1x2 ? percent(metrics.accuracy_1x2) : "—"}</td>
      <td>${metrics.mae_home_goals?.toFixed(3) ?? "—"}</td></tr>`).join("")}
    </tbody></table></div>`;
}

function drawCalibration(series) {
  const canvas = document.querySelector("#calibration-chart");
  const context = canvas.getContext("2d");
  const width = canvas.clientWidth || 700;
  const height = 300;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const padding = { left: 42, right: 20, top: 18, bottom: 38 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  context.strokeStyle = "#536057";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top + chartHeight);
  context.lineTo(padding.left + chartWidth, padding.top);
  context.stroke();
  context.strokeStyle = "#354039";
  context.beginPath();
  context.moveTo(padding.left, padding.top + chartHeight);
  context.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + chartHeight);
  context.stroke();
  const colors = ["#bbf36b", "#77b9ff"];
  series.forEach((item, seriesIndex) => {
    context.fillStyle = colors[seriesIndex];
    item.bins.forEach((bin) => {
      const x = padding.left + bin.mean_predicted * chartWidth;
      const y = padding.top + (1 - bin.observed_frequency) * chartHeight;
      context.beginPath();
      context.arc(x, y, 4, 0, Math.PI * 2);
      context.fill();
    });
    context.fillText(item.name, padding.left + seriesIndex * 125, height - 8);
  });
  context.fillStyle = "#9aa69d";
  context.fillText("Observed", 4, 15);
  context.fillText("Predicted", width - 62, height - 8);
}

document.querySelector("#load-performance").addEventListener("click", async () => {
  const message = document.querySelector("#performance-message");
  const output = document.querySelector("#performance-results");
  message.textContent = "Evaluasi kronologis sedang berjalan…";
  output.replaceChildren();
  try {
    const result = await api(`/api/performance?last_n=${document.querySelector("#evaluation-n").value}`);
    message.textContent = `${result.sample_count} sampel uji · ${result.folds_completed} fold · N=${result.last_n} · temperatur kalibrasi OOF ${result.calibration.temperature.toFixed(2)}`;
    output.innerHTML = `<h3>Keseluruhan</h3>${metricTable(result.overall)}
      ${Object.entries(result.per_competition).map(([code, summary]) => `<h3>${escapeHtml(code)} · ${summary.sample_count} sampel</h3>${metricTable(summary)}`).join("")}
      <p class="message">${result.limitations.map(escapeHtml).join(" ")}</p>`;
    const poissonMetrics = result.overall.poisson;
    const calibrationContainer = document.querySelector("#calibration-container");
    if (poissonMetrics) {
      calibrationContainer.hidden = false;
      drawCalibration([
        { name: "Over 2.5", bins: poissonMetrics.calibration_over_25 },
        { name: "BTTS", bins: poissonMetrics.calibration_btts },
      ]);
    } else {
      calibrationContainer.hidden = true;
    }
  } catch (error) {
    message.textContent = error.message;
  }
});

dayTabs.forEach((tab) => tab.addEventListener("click", () => {
  dayTabs.forEach((item) => {
    item.classList.toggle("active", item === tab);
    item.setAttribute("aria-selected", String(item === tab));
  });
  currentDay = tab.dataset.day;
  loadMatches();
}));
leagueFilter.addEventListener("change", loadMatches);
document.querySelector(".dialog-close").addEventListener("click", () => predictionDialog.close());
predictionDialog.addEventListener("click", (event) => {
  if (event.target === predictionDialog) predictionDialog.close();
});

loadCompetitions().then(loadMatches);
loadTeamDirectory();
