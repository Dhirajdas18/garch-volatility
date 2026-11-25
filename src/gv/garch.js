// Volatility maths in the browser. Mirrors src/gv/models.py; tests compare the two through Node.
// Returns are daily log returns in percent. `params` holds the fitted mu, omega, alpha, gamma,
// beta and the shock moments asym_weight (E[z^2 1(z<0)]), abs_mean (E|z|) and q01 (1% quantile),
// all computed in Python at build time.

const SQRT2_OV_PI = Math.sqrt(2 / Math.PI);
const MODELS = ["GARCH", "GJR", "EGARCH"];

function checkModel(model) {
  if (!MODELS.includes(model)) throw new Error(`model must be one of ${MODELS.join(", ")}`);
}

// The variance before the first day, as arch computes it: a 0.94-weighted average of the first
// 75 squared deviations from the sample mean.
function backcast(r) {
  const mean = r.reduce((a, b) => a + b, 0) / r.length;
  const tau = Math.min(75, r.length);
  let num = 0, den = 0;
  for (let k = 0; k < tau; k++) {
    const w = Math.pow(0.94, k);
    num += w * (r[k] - mean) ** 2; den += w;
  }
  return num / den;
}

function persistence(p, model) {
  checkModel(model);
  if (model === "EGARCH") return p.beta;
  return p.alpha + (p.gamma || 0) * p.asym_weight + p.beta;
}

function halfLife(p, model) {
  const q = persistence(p, model);
  return q > 0 && q < 1 ? Math.log(0.5) / Math.log(q) : Infinity;
}

function longRunVariance(p, model) {
  const q = persistence(p, model);
  if (q >= 1) return NaN;
  if (model === "EGARCH") return Math.exp((p.omega + p.alpha * (p.abs_mean - SQRT2_OV_PI)) / (1 - q));
  return p.omega / (1 - q);
}

// s2[t] for every day, from information up to t - 1 (percent squared).
function conditionalVariance(r, p, model) {
  checkModel(model);
  const n = r.length, s2 = new Array(n), bc = backcast(r);
  const om = p.omega, a = p.alpha, g = p.gamma || 0, b = p.beta;
  if (model === "EGARCH") {
    let lns = om + b * Math.log(bc);
    for (let t = 0; t < n; t++) {
      if (t) {
        const z = (r[t - 1] - p.mu) / Math.sqrt(s2[t - 1]);
        lns = om + a * (Math.abs(z) - SQRT2_OV_PI) + g * z + b * lns;
      }
      s2[t] = Math.exp(lns);
    }
    return s2;
  }
  s2[0] = om + a * bc + g * 0.5 * bc + b * bc;
  for (let t = 1; t < n; t++) {
    const e = r[t - 1] - p.mu;
    s2[t] = om + (a + (e < 0 ? g : 0)) * e * e + b * s2[t - 1];
  }
  return s2;
}

function lastState(r, p, model) {
  const s2 = conditionalVariance(r, p, model);
  return { resid: r[r.length - 1] - p.mu, sigma2: s2[s2.length - 1] };
}

// Variance for each of the next `horizon` days. GARCH and GJR exact; EGARCH the exp of the
// expected log variance (see forecast_variance in models.py).
function forecastVariance(state, p, model, horizon) {
  checkModel(model);
  const e = state.resid, s2 = state.sigma2, out = new Array(horizon);
  const om = p.omega, a = p.alpha, g = p.gamma || 0, b = p.beta;
  if (model === "EGARCH") {
    const z = e / Math.sqrt(s2);
    let lns = om + a * (Math.abs(z) - SQRT2_OV_PI) + g * z + b * Math.log(s2);
    const drift = om + a * (p.abs_mean - SQRT2_OV_PI);
    for (let h = 0; h < horizon; h++) {
      if (h) lns = drift + b * lns;
      out[h] = Math.exp(lns);
    }
    return out;
  }
  out[0] = om + (a + (e < 0 ? g : 0)) * e * e + b * s2;
  const q = persistence(p, model);
  for (let h = 1; h < horizon; h++) out[h] = om + q * out[h - 1];
  return out;
}

// Tomorrow's variance against today's residual, with today's variance at its long-run level.
function newsImpact(p, model, shocks) {
  checkModel(model);
  const lr = longRunVariance(p, model);
  const om = p.omega, a = p.alpha, g = p.gamma || 0, b = p.beta;
  if (model === "EGARCH") {
    return shocks.map((e) => {
      const z = e / Math.sqrt(lr);
      return Math.exp(om + a * (Math.abs(z) - SQRT2_OV_PI) + g * z + b * Math.log(lr));
    });
  }
  return shocks.map((e) => om + (a + (e < 0 ? g : 0)) * e * e + b * lr);
}

// One-day 99% VaR in rupees on `amount`, for log returns in percent.
function valueAtRisk(mu, sigma2, q01, amount = 1e5) {
  return amount * (1 - Math.exp((mu + Math.sqrt(sigma2) * q01) / 100));
}

// Sample standard deviation over the last `window` values ending at each day (null before).
function rollingStd(r, window = 20) {
  return r.map((_, t) => {
    if (t < window - 1) return null;
    let m = 0;
    for (let k = t - window + 1; k <= t; k++) m += r[k];
    m /= window;
    let v = 0;
    for (let k = t - window + 1; k <= t; k++) v += (r[k] - m) ** 2;
    return Math.sqrt(v / (window - 1));
  });
}

if (typeof module !== "undefined") {
  module.exports = {
    backcast, persistence, halfLife, longRunVariance, conditionalVariance, lastState,
    forecastVariance, newsImpact, valueAtRisk, rollingStd,
  };
}
