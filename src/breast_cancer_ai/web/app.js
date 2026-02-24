const pageState = {
  modelInfo: null,
  lastPrediction: null,
  lastFeatures: null,
};

const elements = {
  statusText: document.getElementById("status-text"),
  modelVersion: document.getElementById("model-version"),
  modelThreshold: document.getElementById("model-threshold"),
  predictForm: document.getElementById("predict-form"),
  predictButton: document.getElementById("predict-btn"),
  fillSampleButton: document.getElementById("fill-sample-btn"),
  clearButton: document.getElementById("clear-btn"),
  riskChip: document.getElementById("risk-chip"),
  probabilityValue: document.getElementById("probability-value"),
  predictionLabel: document.getElementById("prediction-label"),
  requestId: document.getElementById("request-id"),
  rawResponse: document.getElementById("raw-response"),
  downloadReportButton: document.getElementById("download-report-btn"),
};

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatFixed(value, digits = 6) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function sanitizeLabel(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function setStatus(text, isError = false) {
  if (!elements.statusText) {
    return;
  }
  elements.statusText.textContent = text;
  elements.statusText.classList.remove("status-ok", "status-error");
  elements.statusText.classList.add(isError ? "status-error" : "status-ok");
}

function sampleValueForFeature(featureName) {
  const key = featureName.toLowerCase();

  if (key.includes("radius_worst")) return 21.4;
  if (key.includes("perimeter_worst")) return 138.7;
  if (key.includes("area_worst")) return 980.0;
  if (key.includes("radius_mean")) return 14.2;
  if (key.includes("perimeter_mean")) return 92.0;
  if (key.includes("area_mean")) return 650.0;
  if (key.includes("texture")) return 18.5;
  if (key.includes("smoothness")) return 0.11;
  if (key.includes("compactness")) return 0.15;
  if (key.includes("concavity")) return 0.14;
  if (key.includes("concave_points")) return 0.08;
  if (key.includes("symmetry")) return 0.22;
  if (key.includes("fractal_dimension")) return 0.063;
  if (key.endsWith("_se") || key.includes("_se")) return 0.85;
  if (key.includes("worst")) return 18.0;
  if (key.includes("radius")) return 14.0;
  if (key.includes("perimeter")) return 95.0;
  if (key.includes("area")) return 600.0;

  return 1.0;
}

function buildFeatureForm(featureNames) {
  if (!elements.predictForm) {
    return;
  }

  elements.predictForm.innerHTML = "";
  const fragment = document.createDocumentFragment();

  featureNames.forEach((featureName) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", `field-${featureName}`);
    label.textContent = sanitizeLabel(featureName);

    const input = document.createElement("input");
    input.id = `field-${featureName}`;
    input.name = featureName;
    input.type = "number";
    input.step = "any";
    input.required = true;
    input.inputMode = "decimal";
    input.placeholder = String(sampleValueForFeature(featureName));

    wrapper.append(label, input);
    fragment.append(wrapper);
  });

  elements.predictForm.append(fragment);
}

function resetPredictionView() {
  if (elements.probabilityValue) elements.probabilityValue.textContent = "-";
  if (elements.predictionLabel) elements.predictionLabel.textContent = "-";
  if (elements.requestId) elements.requestId.textContent = "-";
  if (elements.rawResponse) {
    elements.rawResponse.textContent = "Aucune prediction pour le moment.";
  }
  if (elements.riskChip) {
    elements.riskChip.textContent = "En attente";
    elements.riskChip.className = "risk-chip neutral";
  }
  if (elements.downloadReportButton) {
    elements.downloadReportButton.disabled = true;
  }

  pageState.lastPrediction = null;
  pageState.lastFeatures = null;
}

function readFeaturePayload() {
  if (!elements.predictForm) {
    return {};
  }

  const payload = {};
  const inputs = elements.predictForm.querySelectorAll("input[name]");

  for (const input of inputs) {
    const rawValue = input.value.trim();
    if (!rawValue) {
      throw new Error(`Valeur manquante pour: ${input.name}`);
    }

    const numericValue = Number.parseFloat(rawValue);
    if (!Number.isFinite(numericValue)) {
      throw new Error(`Valeur invalide pour: ${input.name}`);
    }

    payload[input.name] = numericValue;
  }

  return payload;
}

function updateRiskChip(probability, threshold) {
  if (!elements.riskChip) {
    return;
  }

  if (probability >= threshold) {
    elements.riskChip.textContent = "Risque eleve - orientation oncologie";
    elements.riskChip.className = "risk-chip high";
    return;
  }

  elements.riskChip.textContent = "Risque faible";
  elements.riskChip.className = "risk-chip low";
}

function renderPrediction(result, features) {
  if (elements.probabilityValue) {
    elements.probabilityValue.textContent = formatPercent(result.probability_malignant);
  }
  if (elements.predictionLabel) {
    elements.predictionLabel.textContent = result.prediction_label;
  }
  if (elements.requestId) {
    elements.requestId.textContent = result.request_id;
  }
  if (elements.rawResponse) {
    elements.rawResponse.textContent = JSON.stringify(result, null, 2);
  }

  updateRiskChip(result.probability_malignant, result.threshold);

  pageState.lastPrediction = result;
  pageState.lastFeatures = features;

  if (elements.downloadReportButton) {
    elements.downloadReportButton.disabled = false;
  }
}

function buildMedicalReport(prediction, features) {
  const generatedAt = new Date();
  const sortedFeatures = Object.entries(features).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  const featureLines = sortedFeatures
    .map(([name, value]) => `- ${name}: ${value}`)
    .join("\n");

  const triageConclusion =
    prediction.prediction === 1
      ? "Suspicion de malignite elevee: bilan oncologique prioritaire recommande."
      : "Suspicion de malignite faible: poursuivre la filiere clinique standard.";

  return [
    "COMPTE-RENDU MEDICAL - TRIAGE ALGORITHMIQUE",
    "Service: OncoBreast AI Console (Decision support)",
    "",
    `Date generation: ${generatedAt.toISOString()}`,
    `Model version: ${prediction.model_version}`,
    `Request ID: ${prediction.request_id}`,
    "",
    "Resultat inference:",
    `- Probabilite de malignite: ${formatPercent(prediction.probability_malignant)}`,
    `- Decision systeme: ${prediction.prediction_label}`,
    `- Seuil clinique applique: ${formatFixed(prediction.threshold, 6)}`,
    "",
    "Conclusion de triage:",
    `- ${triageConclusion}`,
    "",
    "Variables cliniques exploitees:",
    featureLines,
    "",
    "Avertissement:",
    "Ce document est un support d'aide a la decision.",
    "Il ne remplace pas le diagnostic medical, la biopsie, ni l'avis specialise.",
  ].join("\n");
}

function downloadMedicalReport() {
  if (!pageState.lastPrediction || !pageState.lastFeatures) {
    return;
  }

  const reportText = buildMedicalReport(pageState.lastPrediction, pageState.lastFeatures);
  const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const now = new Date();
  const timestamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(
    now.getDate(),
  ).padStart(2, "0")}_${String(now.getHours()).padStart(2, "0")}${String(
    now.getMinutes(),
  ).padStart(2, "0")}${String(now.getSeconds()).padStart(2, "0")}`;

  const link = document.createElement("a");
  link.href = url;
  link.download = `compte_rendu_oncobreast_${timestamp}.txt`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function fetchJson(url, init = undefined) {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText} - ${body}`);
  }
  return response.json();
}

async function loadMetadata() {
  const [health, modelInfo] = await Promise.all([
    fetchJson("/health"),
    fetchJson("/model_info"),
  ]);

  pageState.modelInfo = modelInfo;
  if (elements.modelVersion) elements.modelVersion.textContent = health.model_version;
  if (elements.modelThreshold) {
    elements.modelThreshold.textContent = formatFixed(Number(modelInfo.threshold), 6);
  }

  buildFeatureForm(modelInfo.feature_names);
  setStatus("API operationnelle");
}

async function runPrediction() {
  if (!elements.predictButton) {
    return;
  }

  try {
    const features = readFeaturePayload();
    elements.predictButton.disabled = true;
    elements.predictButton.textContent = "Prediction en cours...";

    const response = await fetchJson("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });

    renderPrediction(response, features);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Erreur inconnue";
    if (elements.rawResponse) {
      elements.rawResponse.textContent = message;
    }
    if (elements.riskChip) {
      elements.riskChip.textContent = "Erreur";
      elements.riskChip.className = "risk-chip neutral";
    }
  } finally {
    elements.predictButton.disabled = false;
    elements.predictButton.textContent = "Lancer la prediction";
  }
}

function fillSampleForm() {
  if (!elements.predictForm) {
    return;
  }

  const inputs = elements.predictForm.querySelectorAll("input[name]");
  inputs.forEach((input) => {
    input.value = String(sampleValueForFeature(input.name));
  });
}

function clearForm() {
  if (!elements.predictForm) {
    return;
  }

  const inputs = elements.predictForm.querySelectorAll("input[name]");
  inputs.forEach((input) => {
    input.value = "";
  });
  resetPredictionView();
}

async function init() {
  resetPredictionView();

  elements.predictButton?.addEventListener("click", runPrediction);
  elements.fillSampleButton?.addEventListener("click", fillSampleForm);
  elements.clearButton?.addEventListener("click", clearForm);
  elements.downloadReportButton?.addEventListener("click", downloadMedicalReport);

  try {
    await loadMetadata();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Erreur API";
    setStatus("API indisponible", true);
    if (elements.rawResponse) {
      elements.rawResponse.textContent = message;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  void init();
});
