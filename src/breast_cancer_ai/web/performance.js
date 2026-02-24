const perfElements = {
  modelVersion: document.getElementById("perf-model-version"),
  updated: document.getElementById("perf-updated"),
  source: document.getElementById("perf-source"),
  badge: document.getElementById("perf-badge"),
  error: document.getElementById("perf-error"),
  kpiSensitivity: document.getElementById("kpi-test-sensitivity"),
  kpiSpecificity: document.getElementById("kpi-test-specificity"),
  kpiRocAuc: document.getElementById("kpi-test-roc-auc"),
  kpiPrAuc: document.getElementById("kpi-test-pr-auc"),
  kpiAccuracy: document.getElementById("kpi-test-accuracy"),
  kpiThreshold: document.getElementById("kpi-threshold"),
};

function asObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : Number.NaN;
}

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatFloat(value, digits = 4) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function setKpi(element, value, formatter) {
  if (!element) {
    return;
  }
  element.textContent = formatter(value);
}

function getMetrics(report, key) {
  return asObject(report[key]);
}

function cohortSeries(report, metricKey) {
  const cohorts = [
    { label: "CV OOF", key: "cv_oof_metrics" },
    { label: "Train+Val", key: "trainval_metrics" },
    { label: "Test externe", key: "test_metrics" },
  ];

  const labels = [];
  const values = [];

  cohorts.forEach((cohort) => {
    const metrics = getMetrics(report, cohort.key);
    labels.push(cohort.label);
    values.push(asNumber(metrics[metricKey]));
  });

  return { labels, values };
}

function animatedLinePlot(targetId, traces, layout) {
  const target = document.getElementById(targetId);
  if (!target) {
    return;
  }

  const config = { responsive: true, displayModeBar: false };
  const seededTraces = traces.map((trace) => ({
    ...trace,
    x: trace.x.slice(0, 1),
    y: trace.y.slice(0, 1),
    marker: {
      ...(trace.marker ?? {}),
      size: 6,
    },
  }));

  Plotly.newPlot(target, seededTraces, layout, config)
    .then(() => {
      const maxPoints = Math.max(...traces.map((trace) => trace.x.length), 1);
      let index = 2;

      const timer = window.setInterval(() => {
        const nextTraces = traces.map((trace) => ({
          ...trace,
          x: trace.x.slice(0, index),
          y: trace.y.slice(0, index),
          marker: {
            ...(trace.marker ?? {}),
            size: 12,
          },
        }));

        void Plotly.animate(
          target,
          { data: nextTraces },
          {
            transition: { duration: 520, easing: "cubic-in-out" },
            frame: { duration: 520, redraw: true },
          },
        );

        if (index >= maxPoints) {
          window.clearInterval(timer);
          let pulseSize = 12;
          let growing = true;
          let cycles = 0;
          const pulseTimer = window.setInterval(() => {
            pulseSize += growing ? 1 : -1;
            if (pulseSize >= 15) {
              growing = false;
            }
            if (pulseSize <= 11) {
              growing = true;
              cycles += 1;
            }
            const pulsePayload = traces.map((trace) => ({
              marker: {
                ...(trace.marker ?? {}),
                size: pulseSize,
              },
            }));
            void Plotly.restyle(target, pulsePayload);
            if (cycles >= 2) {
              window.clearInterval(pulseTimer);
              const settlePayload = traces.map((trace) => ({
                marker: {
                  ...(trace.marker ?? {}),
                  size: trace.marker?.size ?? 11,
                },
              }));
              void Plotly.restyle(target, settlePayload);
            }
          }, 120);
        }
        index += 1;
      }, 760);
    })
    .catch((error) => {
      console.error("Plotly render error", error);
    });
}

function renderKpis(report) {
  const testMetrics = getMetrics(report, "test_metrics");

  setKpi(perfElements.kpiSensitivity, asNumber(testMetrics.sensitivity), formatPercent);
  setKpi(perfElements.kpiSpecificity, asNumber(testMetrics.specificity), formatPercent);
  setKpi(perfElements.kpiRocAuc, asNumber(testMetrics.roc_auc), formatFloat);
  setKpi(perfElements.kpiPrAuc, asNumber(testMetrics.pr_auc), formatFloat);
  setKpi(perfElements.kpiAccuracy, asNumber(testMetrics.accuracy), formatPercent);
  setKpi(perfElements.kpiThreshold, asNumber(report.threshold), (value) => formatFloat(value, 6));
}

function renderAucChart(report) {
  const roc = cohortSeries(report, "roc_auc");
  const pr = cohortSeries(report, "pr_auc");

  const traces = [
    {
      x: roc.labels,
      y: roc.values,
      type: "scatter",
      mode: "lines+markers",
      name: "ROC AUC",
      line: { color: "#0b8a76", width: 4, shape: "spline" },
      marker: { size: 11, color: "#0b8a76" },
      hovertemplate: "%{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
    },
    {
      x: pr.labels,
      y: pr.values,
      type: "scatter",
      mode: "lines+markers",
      name: "PR AUC",
      line: { color: "#f97316", width: 4, shape: "spline" },
      marker: { size: 11, color: "#f97316" },
      hovertemplate: "%{x}<br>PR AUC: %{y:.4f}<extra></extra>",
    },
  ];

  const layout = {
    margin: { t: 24, r: 18, b: 56, l: 52 },
    legend: { orientation: "h", y: 1.2 },
    yaxis: { range: [0.85, 1.01], title: "AUC" },
    xaxis: { title: "Cohorte" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.85)",
  };

  animatedLinePlot("auc-curve", traces, layout);
}

function renderSensitivitySpecificityChart(report) {
  const sensitivity = cohortSeries(report, "sensitivity");
  const specificity = cohortSeries(report, "specificity");

  const thresholdDetails = asObject(report.threshold_details);
  const targetSensitivity = asNumber(thresholdDetails.target_sensitivity);

  const traces = [
    {
      x: sensitivity.labels,
      y: sensitivity.values,
      type: "scatter",
      mode: "lines+markers",
      name: "Sensibilite",
      line: { color: "#b91c1c", width: 4, shape: "spline" },
      marker: { size: 11, color: "#b91c1c" },
      hovertemplate: "%{x}<br>Sensibilite: %{y:.3f}<extra></extra>",
    },
    {
      x: specificity.labels,
      y: specificity.values,
      type: "scatter",
      mode: "lines+markers",
      name: "Specificite",
      line: { color: "#1d4ed8", width: 4, shape: "spline" },
      marker: { size: 11, color: "#1d4ed8" },
      hovertemplate: "%{x}<br>Specificite: %{y:.3f}<extra></extra>",
    },
  ];

  if (Number.isFinite(targetSensitivity)) {
    traces.push({
      x: sensitivity.labels,
      y: sensitivity.labels.map(() => targetSensitivity),
      type: "scatter",
      mode: "lines",
      name: "Cible sensibilite",
      line: { color: "#6b7280", width: 2, dash: "dash" },
      hovertemplate: "Cible sensibilite: %{y:.3f}<extra></extra>",
    });
  }

  const layout = {
    margin: { t: 24, r: 18, b: 56, l: 52 },
    legend: { orientation: "h", y: 1.2 },
    yaxis: { range: [0.85, 1.01], title: "Score" },
    xaxis: { title: "Cohorte" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.85)",
  };

  animatedLinePlot("sens-spec-curve", traces, layout);
}

function renderConfusionMatrix(report) {
  const test = getMetrics(report, "test_metrics");
  const tn = Number(asNumber(test.tn) || 0);
  const fp = Number(asNumber(test.fp) || 0);
  const fn = Number(asNumber(test.fn) || 0);
  const tp = Number(asNumber(test.tp) || 0);

  const z = [
    [tn, fp],
    [fn, tp],
  ];

  const cellText = [
    [
      `VN: ${tn}<br>Absence de malignite correctement exclue`,
      `FP: ${fp}<br>Suspicion de malignite non confirmee`,
    ],
    [
      `FN: ${fn}<br>Malignite non detectee (risque clinique)`,
      `VP: ${tp}<br>Malignite detectee et orientee`,
    ],
  ];

  const trace = {
    z,
    type: "heatmap",
    colorscale: [
      [0.0, "#fef3c7"],
      [0.5, "#fdba74"],
      [1.0, "#b91c1c"],
    ],
    text: cellText,
    texttemplate: "%{text}",
    hovertemplate: "<b>%{y}</b><br>%{x}<br>Effectif: %{z}<extra></extra>",
    x: [
      "Decision: Benin non suspect",
      "Decision: Malignite suspectee",
    ],
    y: [
      "Reference anatomo-pathologique: Benin",
      "Reference anatomo-pathologique: Malin",
    ],
  };

  const layout = {
    margin: { t: 24, r: 12, b: 92, l: 180 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.85)",
  };

  void Plotly.newPlot("confusion-matrix", [trace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

function renderFoldStability(report) {
  const foldsReport = Array.isArray(report.cv_folds_report) ? report.cv_folds_report : [];
  if (foldsReport.length === 0) {
    return;
  }

  const labels = foldsReport.map((foldEntry) => `Fold ${foldEntry.fold}`);
  const sensitivityValues = foldsReport.map((foldEntry) =>
    asNumber(asObject(foldEntry.fold_metrics).sensitivity),
  );
  const specificityValues = foldsReport.map((foldEntry) =>
    asNumber(asObject(foldEntry.fold_metrics).specificity),
  );
  const aucValues = foldsReport.map((foldEntry) =>
    asNumber(asObject(foldEntry.fold_metrics).roc_auc),
  );

  const traces = [
    {
      x: labels,
      y: sensitivityValues,
      type: "scatter",
      mode: "lines+markers",
      name: "Sensibilite fold",
      line: { color: "#b91c1c", width: 3 },
      marker: { size: 9 },
    },
    {
      x: labels,
      y: specificityValues,
      type: "scatter",
      mode: "lines+markers",
      name: "Specificite fold",
      line: { color: "#1d4ed8", width: 3 },
      marker: { size: 9 },
    },
    {
      x: labels,
      y: aucValues,
      type: "scatter",
      mode: "lines+markers",
      name: "ROC AUC fold",
      line: { color: "#0b8a76", width: 3 },
      marker: { size: 9 },
    },
  ];

  const layout = {
    margin: { t: 24, r: 18, b: 56, l: 52 },
    legend: { orientation: "h", y: 1.2 },
    yaxis: { range: [0.85, 1.01], title: "Score" },
    xaxis: { title: "Fold" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.85)",
  };

  animatedLinePlot("fold-stability", traces, layout);
}

function showError(message) {
  if (perfElements.badge) {
    perfElements.badge.textContent = "Erreur";
  }
  if (perfElements.error) {
    perfElements.error.textContent = message;
  }
}

async function fetchPerformancePayload() {
  const response = await fetch("/performance");
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText} - ${body}`);
  }
  return response.json();
}

async function initPerformanceDashboard() {
  try {
    const payload = await fetchPerformancePayload();
    const report = asObject(payload.report);

    if (perfElements.modelVersion) {
      perfElements.modelVersion.textContent = String(report.model_version ?? "-");
    }
    if (perfElements.updated) {
      perfElements.updated.textContent = String(report.created_at_utc ?? "-");
    }
    if (perfElements.source) {
      perfElements.source.textContent = String(payload.metrics_path ?? "-");
    }
    if (perfElements.badge) {
      perfElements.badge.textContent = "Metriques chargees";
    }
    if (perfElements.error) {
      perfElements.error.textContent = "";
    }

    renderKpis(report);
    renderAucChart(report);
    renderSensitivitySpecificityChart(report);
    renderConfusionMatrix(report);
    renderFoldStability(report);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Erreur de chargement du dashboard";
    showError(message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  void initPerformanceDashboard();
});
