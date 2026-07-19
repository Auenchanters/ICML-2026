"""Fill the landscape_4col_neutral template with the FORS reproduction content."""
import re
from pathlib import Path

T = Path("tools/posterly/templates/landscape_4col_neutral.html").read_text(encoding="utf-8")

# --- design tokens: derive accent from a diffusion/sampling-theory blue-teal ---
T = T.replace("--accent:        #2D5F8B;", "--accent:        #1F6F6B;")
T = T.replace("--accent-deep:   #1F4566;", "--accent-deep:   #14514E;")
T = T.replace("--accent-light:  #E8F1F8;", "--accent-light:  #E4F1F0;")
T = T.replace("--accent-soft:   #D7E5F0;", "--accent-soft:   #CCE5E3;")

# --- HEADER ---
T = T.replace(
'''    <div class="venue-badge">
      <div class="vb-venue">VENUE</div>
      <div class="vb-year">YEAR</div>
      <div class="vb-tag">POSTER</div>
    </div>''',
'''    <div class="venue-badge">
      <div class="vb-venue">ICML</div>
      <div class="vb-year">2026</div>
      <div class="vb-tag">REPRO</div>
    </div>''')

T = T.replace(
'''      <h1 class="title">POSTER TITLE: <span class="accent">Subtitle Keyword</span></h1>
      <div class="subtitle">One-sentence tagline that explains the work in plain English.</div>
      <div class="authors-line">
        <span class="author">Author One</span> &middot;
        <span class="author">Author Two</span> &middot;
        <span class="author">Author Three<sup>&#9993;</sup></span>
        <span class="aff">Lab / Department &middot; Institution &middot; City, Country</span>
      </div>''',
'''      <h1 class="title">High-Accuracy Sampling for Diffusion Models &amp; Log-Concave Distributions <span class="accent">[ An Independent Reproduction ]</span></h1>
      <div class="subtitle">All 5 claims of a theory-only ICML 2026 paper, verified from scratch by deterministic quadrature &mdash; zero Monte-Carlo noise in the headline results.</div>
      <div class="authors-line">
        <span class="author">Agent reproduction (Claude Code)<sup>&#9993;</sup></span>
        <span class="aff">&middot; Paper #10218 (OpenReview GW3umRqsZZ, arXiv:2602.01338) &middot; HF &times; alphaXiv &times; Trackio challenge</span>
      </div>''')

# delete empty logo-slot AND the qr-block (no real URL yet) — the whole
# right-block becomes empty, so drop it entirely for a clean centered header.
T = re.sub(r'<!-- RIGHT: optional logo.*?<div class="right-block">.*?</div>\s*</div>\s*</header>',
           '</header>', T, flags=re.DOTALL, count=1)

# --- FRAMEWORK BANNER ---
T = T.replace(
'''    <div class="fb-text">
      <span class="fb-label">Framework</span>
      &nbsp;<strong>Method name</strong> in one sentence — what it does, what it improves, what's new.
    </div>
    <div class="banner-stats">
      <div class="bs-item"><div class="bs-num">N&times;</div><div class="bs-label">headline<br>improvement</div></div>
      <div class="bs-item"><div class="bs-num">SOTA</div><div class="bs-label">benchmark<br>positioning</div></div>
      <div class="bs-item"><div class="bs-num">key</div><div class="bs-label">property<br>(e.g. proxy-free)</div></div>
      <div class="bs-item"><div class="bs-num">tag</div><div class="bs-label">domain<br>(e.g. offline RL)</div></div>
    </div>''',
'''    <div class="fb-text">
      <span class="fb-label">FORS</span>
      &nbsp;<strong>First-Order Rejection Sampling</strong> &mdash; a Bernoulli-factory sampler that tilts a proposal by <em>unbiased estimates</em> of a log-density, needing no density evaluations. Applied to diffusion backward kernels it reaches error &delta; in <strong>polylog(1/&delta;)</strong> steps, an exponential improvement over the poly(1/&delta;) of prior samplers.
    </div>
    <div class="banner-stats">
      <div class="bs-item"><div class="bs-num">5&#47;5</div><div class="bs-label">claims<br>reproduced</div></div>
      <div class="bs-item"><div class="bs-num">log&sup2;&#8901;&#8308;</div><div class="bs-label">step-count degree<br>vs theory log&sup3;</div></div>
      <div class="bs-item"><div class="bs-num">0</div><div class="bs-label">Monte-Carlo noise<br>(certified)</div></div>
      <div class="bs-item"><div class="bs-num">~&#36;3</div><div class="bs-label">GPU cost<br>of &#36;20 credit</div></div>
    </div>''')

# --- COLUMN 1: Motivation + Claim 1 ---
T = T.replace(
'''      <div class="card highlight" data-measure-role="card">
        <div class="section-title"><span class="num">1</span><span class="st-text">Motivation</span></div>
        <p class="body-text">
          TODO: 2-3 sentences on the problem. Use <span class="keyword">accent color</span> for terms you want the eye to land on.
        </p>
        <ul class="mt-3 fs-4">
          <li>TODO: pain point 1.</li>
          <li>TODO: pain point 2.</li>
        </ul>
        <div class="callout mt-4">
          <strong>Q:</strong> TODO: the question your paper answers.
        </div>
      </div>

      <div class="card" data-measure-role="card">
        <div class="section-title"><span class="num">2</span><span class="st-text">Key Insight</span></div>
        <p class="body-text">
          TODO: 2-3 sentences on the central insight.
        </p>
        <div class="figure mt-3">
          <!-- Key-insight figure from the paper:
               <img src="assets/paper_figures/key-figure.png" data-source="paper"
                    data-asset-id="key-figure" class="w-95"> -->
          <div class="caption">
            TODO: caption explaining (a) / (b) / (c) panels in one line each.
          </div>
        </div>
      </div>''',
'''      <div class="card highlight" data-measure-role="card" data-logbook-target="plan-provenance">
        <div class="section-title"><span class="num">0</span><span class="st-text">A theory paper, reproduced</span></div>
        <p class="body-text">
          The paper has <span class="keyword">no code and no experiments</span> (&ldquo;experimental evaluation &hellip; left for future work&rdquo;). So a faithful <span class="keyword">independent implementation</span> verified against closed-form ground truth <em>is</em> the reproduction.
        </p>
        <ul class="mt-3 fs-4">
          <li>Targets with <strong>exact scores</strong> (&epsilon;<sub>score</sub> = 0).</li>
          <li>Error certified by <strong>deterministic quadrature</strong> through the paper&rsquo;s own chain-rule decomposition (Sec.&nbsp;F.2).</li>
        </ul>
        <div class="callout mt-4">
          <strong>Method:</strong> execute the proof numerically &mdash; certify each per-step KL to float64 precision.
        </div>
      </div>

      <div class="card" data-measure-role="card" data-logbook-target="claim-1-fors-exactness-theorem-3-1">
        <div class="section-title"><span class="num">1</span><span class="st-text">FORS exactness &amp; draw bound (Thm 3.1)&nbsp;<span class="key-mark">&#10003;</span></span></div>
        <p class="body-text">
          FORS output density is exactly q&middot;e<sup>E[W|x]</sup>; per-x acceptance is exactly e<sup>E[W|x]&minus;B</sup>. Both verified as <strong>exact identities</strong>.
        </p>
        <div class="keybox mt-3">
          <div class="kb-item"><div class="kb-num">p=0.20</div><div class="kb-label">exact-law GOF<br>at n=10&#8311;</div></div>
          <div class="kb-item"><div class="kb-num">4 sig figs</div><div class="kb-label">mean draws<br>= 2B/A</div></div>
          <div class="kb-item"><div class="kb-num">&#8804;50&times;</div><div class="kb-label">slack under<br>3Be&sup2;&#7495;log(2/&delta;)</div></div>
        </div>
        <div class="callout mt-3">
          <strong>NC-0:</strong> a biased estimator is rejected at <strong>p &#8776; 0</strong> &mdash; the test has power.
        </div>
      </div>

      <div class="card" data-measure-role="card" data-logbook-target="claim-1-fors-exactness-theorem-3-1">
        <div class="section-title"><span class="num">A1</span><span class="st-text">How FORS works (Algorithm 1)</span></div>
        <p class="body-text">
          Draw x&nbsp;~&nbsp;q and J&nbsp;~&nbsp;Poisson(2B); accept x with probability &prod;<sub>j&#8804;J</sub>(B+W<sub>j</sub>)/2B, using only i.i.d. unbiased draws W<sub>j</sub> of the tilt.
        </p>
        <div class="eqn">
          <span class="label">Bernoulli-factory identity</span>
          $$e^{\\mathbb{E} W} \\;=\\; e^{-1}\\,\\mathbb{E}\\!\\Big[\\textstyle\\prod_{j\\le J}\\tfrac{1+W_j}{2}\\Big],\\quad J\\sim\\mathrm{Poisson}(2)$$
        </div>
      </div>''')

# --- COLUMN 2: Claim 2 (polylog) with money1 ---
T = T.replace(
'''      <div class="card" data-measure-role="card">
        <div class="section-title"><span class="num">3</span><span class="st-text">Method Step 1</span></div>
        <p class="body-text">TODO: how step 1 works.</p>
        <div class="eqn">
          <span class="label">Core equation</span>
          $$f(x) \\;=\\; \\text{TODO}$$
        </div>
        <p class="body-text">TODO: interpretation.</p>
      </div>

      <div class="card highlight" data-measure-role="card">
        <div class="section-title"><span class="num">4</span><span class="st-text">Method Step 2&nbsp;<span class="key-mark">&#9733; KEY</span></span></div>
        <p class="body-text">TODO: how step 2 works.</p>
        <div class="eqn eqn--large">
          <span class="label">Step 2 equation</span>
          $$g(x) \\;=\\; \\text{TODO}$$
        </div>
        <div class="callout gold">
          <strong>Theorem.</strong> TODO: one-sentence statement, with $\\varepsilon$ bounds if applicable.
        </div>
      </div>''',
'''      <div class="card highlight" data-measure-role="card" data-logbook-target="claim-2-polylog-minimal-assumptions">
        <div class="section-title"><span class="num">2</span><span class="st-text">polylog(1/&delta;) steps (Thm 4.1)&nbsp;<span class="key-mark">&#9733; Headline</span></span></div>
        <p class="body-text fs-3 text-secondary mb-1">
          Bimodal mixture, exact scores, Cor-4.4 schedule; certified ladder &delta;&nbsp;=&nbsp;10&#8315;&sup1;&#8230;10&#8315;&#8312;.
        </p>
        <div class="figure">
          <img src="assets/money1.png" data-source="reproduction" data-asset-id="money1" class="w-100">
          <div class="caption fs-2">
            <strong>FORS</strong> step count is flat-in-&delta; (polylog); <strong>DDPM</strong> baseline (same proposal, no corrector) climbs as a power law.
          </div>
        </div>
        <div class="keybox">
          <div class="kb-item"><div class="kb-num">2.24</div><div class="kb-label">degree in log(1/&delta;)<br>R&sup2;=0.997 (&#8804;3)</div></div>
          <div class="kb-item"><div class="kb-num">&#8804;3e&minus;10</div><div class="kb-label">certified KL<br>every &delta;</div></div>
          <div class="kb-item"><div class="kb-num">2.06</div><div class="kb-label">DDPM power<br>R&sup2;=0.9995</div></div>
        </div>
      </div>

      <div class="card" data-measure-role="card" data-logbook-target="negative-controls-robustness">
        <div class="section-title"><span class="num">2b</span><span class="st-text">Robustness &amp; condition necessity</span></div>
        <ul class="mt-1 fs-4">
          <li><strong>Arm C:</strong> KL floor = 0.59&times;&Sigma;&eta;&epsilon;&sup2; &mdash; exact &epsilon;&sup2; scaling.</li>
          <li><strong>NC-2:</strong> condition (16) at 0.03&times; blows per-step KL to <strong>16,000&times;</strong> target.</li>
          <li><strong>Arm B:</strong> sampling hits the noise floor; O(K) queries.</li>
        </ul>
        <p class="body-text fs-4 mt-2">d-budget verified <strong>sufficient at every d</strong>.</p>
      </div>''')

# --- COLUMN 3: Claim 4 (intrinsic) with money2 + Claim 3 (Lipschitz) ---
T = T.replace(
'''      <div class="card highlight" data-measure-role="card">
        <div class="section-title"><span class="num">5</span><span class="st-text">Main Result&nbsp;<span class="key-mark">&#9733; Headline</span></span></div>
        <p class="body-text fs-3 text-secondary mb-1">
          TODO: setup line — dataset, # samples, baselines.
        </p>
        <div class="figure">
          <!-- Main-result figure from the paper:
               <img src="assets/paper_figures/main-result.png" data-source="paper"
                    data-asset-id="main-result" class="w-100"> -->
          <div class="caption fs-2">
            <strong>Left:</strong> TODO. <strong>Right:</strong> TODO.
          </div>
        </div>
        <div class="keybox">
          <div class="kb-item"><div class="kb-num">A</div><div class="kb-label">stat 1<br>caption</div></div>
          <div class="kb-item"><div class="kb-num">B</div><div class="kb-label">stat 2<br>caption</div></div>
          <div class="kb-item"><div class="kb-num">C</div><div class="kb-label">stat 3<br>caption</div></div>
        </div>
      </div>

      <div class="card" data-measure-role="card">
        <div class="section-title"><span class="num">6</span><span class="st-text">Robustness / Secondary</span></div>
        <p class="body-text">TODO: 1-2 sentences on robustness, ablation, or secondary axis.</p>
        <ul class="mt-3">
          <li><strong>Setting A</strong> — TODO.</li>
          <li><strong>Setting B</strong> — TODO.</li>
        </ul>
        <div class="callout mt-3">
          TODO: bottom-line takeaway from the robustness study.
        </div>
      </div>''',
'''      <div class="card highlight" data-measure-role="card" data-logbook-target="claim-4-intrinsic-dimension">
        <div class="section-title"><span class="num">4</span><span class="st-text">Intrinsic dimension d&#9733; (Thm 4.6)&nbsp;<span class="key-mark">&#9733;</span></span></div>
        <p class="body-text fs-3 text-secondary mb-1">
          Subspace mixture (d&#9733;&#8776;2) embedded in R&#7496;, ambient d up to 512.
        </p>
        <div class="figure">
          <img src="assets/money2.png" data-source="reproduction" data-asset-id="money2" class="w-100">
          <div class="caption fs-2">
            Critical schedule stiffness G&#9733; is <strong>flat in ambient d</strong> for low-intrinsic-dimension targets, but grows linearly for full-rank ones.
          </div>
        </div>
        <div class="keybox">
          <div class="kb-item"><div class="kb-num">d&#8304;&#8901;&#8304;&sup2;</div><div class="kb-label">subspace G&#9733;<br>(flat)</div></div>
          <div class="kb-item"><div class="kb-num">4 digits</div><div class="kb-label">E[tr &nabla;m] flat<br>to d=512</div></div>
          <div class="kb-item"><div class="kb-num">d=128</div><div class="kb-label">e2e at noise<br>floor, acc 0.37</div></div>
        </div>
      </div>

      <div class="card" data-measure-role="card" data-logbook-target="claim-3-non-uniform-lipschitz">
        <div class="section-title"><span class="num">3</span><span class="st-text">Non-uniform Lipschitz &#8730;(dL) (Thm 4.4)&nbsp;<span class="key-mark">&#10003;</span></span></div>
        <p class="body-text">
          On log-concave targets (L<sub>op</sub>&#8801;1) the measured critical stiffness scales as <strong>&#8730;d, not d</strong> &mdash; the refinement, in both VP and VE.
        </p>
        <ul class="mt-2 fs-4">
          <li>G&#9733; &#8733; d<sup>0.41</sup> (VP) and d<sup>0.41</sup> (VE) &mdash; the &#8730;d law, within 0.5&plusmn;0.1.</li>
          <li>&#8214;&nabla;m<sub>&tau;</sub>&#8214;<sub>op</sub> &#8804; 1 exact; Prop 4.7 constant C &#8712; [0.46, 1.09].</li>
          <li><strong>NC-3:</strong> high-&#8214;&nabla;m&#8214; states are exponentially rare under p<sub>&tau;</sub>.</li>
        </ul>
      </div>''')

# --- COLUMN 4: Claim 5 with money3 + verdict table ---
T = T.replace(
'''      <div class="card highlight" data-measure-role="card">
        <div class="section-title"><span class="num">7</span><span class="st-text">Benchmark Table</span></div>
        <p class="body-text fs-2 text-secondary mb-1">
          TODO: setup line — samples, seeds, metric description.
        </p>
        <table class="result-table">
          <thead>
            <tr>
              <th class="method">Method</th>
              <th>Metric A &#8593;</th>
              <th>Metric B &#8593;</th>
              <th>Cost &#8595;</th>
            </tr>
          </thead>
          <tbody>
            <tr class="reference"><td class="method"><em>Reference</em></td><td>x.xx</td><td>x.xx</td><td>x.xx</td></tr>
            <tr><td class="method">Baseline 1</td><td>x.xx</td><td>x.xx</td><td>x.xx</td></tr>
            <tr><td class="method">Baseline 2</td><td>x.xx</td><td>x.xx</td><td>x.xx</td></tr>
            <tr class="ours"><td class="method">Ours</td><td class="best">x.xx</td><td class="best">x.xx</td><td class="best">x.xx</td></tr>
          </tbody>
        </table>
        <p class="body-text mt-2 fs-3">
          TODO: one-sentence interpretation.
        </p>
      </div>

      <div class="card" data-measure-role="card">
        <div class="section-title"><span class="num">8</span><span class="st-text">Diversity / Extra View</span></div>
        <p class="body-text">TODO: 1-sentence framing.</p>
        <div class="figure">
          <!-- Secondary figure from the paper:
               <img src="assets/paper_figures/secondary-figure.png" data-source="paper"
                    data-asset-id="secondary-figure" class="w-100"> -->
          <div class="caption fs-2">TODO: caption.</div>
        </div>
      </div>''',
'''      <div class="card highlight" data-measure-role="card" data-logbook-target="claim-5-log-concave-sampling-section-5">
        <div class="section-title"><span class="num">5</span><span class="st-text">Log-concave, first-order only (Sec. 5)&nbsp;<span class="key-mark">&#9733;</span></span></div>
        <p class="body-text fs-3 text-secondary mb-1">
          Proximal sampler with FORS-RGO on f = x&sup2;/2 + log cosh 2x.
        </p>
        <div class="figure">
          <img src="assets/money3.png" data-source="reproduction" data-asset-id="money3" class="w-100">
          <div class="caption fs-2">
            <strong>ULA</strong> plateaus at its O(h) discretization-bias wall for every step size; <strong>FORS-proximal</strong> drives &chi;&sup2; down to the machine floor.
          </div>
        </div>
        <div class="keybox">
          <div class="kb-item"><div class="kb-num">0.92</div><div class="kb-label">query degree<br>R&sup2;=0.9997 (&#8804;2)</div></div>
          <div class="kb-item"><div class="kb-num">1.3e&minus;11</div><div class="kb-label">&chi;&sup2; floor<br>(&#8804;1e&minus;10)</div></div>
          <div class="kb-item"><div class="kb-num">&rarr;0</div><div class="kb-label">RGO &chi;&sup2; to<br>machine zero</div></div>
        </div>
      </div>

      <div class="card" data-measure-role="card" data-logbook-target="conclusion">
        <div class="section-title"><span class="num">6</span><span class="st-text">Verdict &amp; provenance</span></div>
        <table class="result-table">
          <thead><tr><th class="method">Claim</th><th>Measured</th><th>Theory</th></tr></thead>
          <tbody>
            <tr class="ours"><td class="method">1 &mdash; FORS Thm 3.1</td><td class="best">exact</td><td>identity</td></tr>
            <tr class="ours"><td class="method">2 &mdash; polylog</td><td class="best">deg 2.24</td><td>&#8804;3</td></tr>
            <tr class="ours"><td class="method">3 &mdash; &#8730;(dL)</td><td class="best">d&#8304;&#8901;&#8308;&sup1;</td><td>0.5</td></tr>
            <tr class="ours"><td class="method">4 &mdash; d&#9733;</td><td class="best">flat</td><td>d&#9733;&#8810;d</td></tr>
            <tr class="ours"><td class="method">5 &mdash; log-concave</td><td class="best">deg 0.92</td><td>&#8804;2</td></tr>
          </tbody>
        </table>
      </div>''')

# --- TAKEAWAYS STRIP ---
T = T.replace(
'''    <div class="ts-title"><span class="num">9</span> Takeaways</div>
    <div class="ts-item"><span class="ts-key">Idea.</span><span class="ts-text">TODO: 1-line.</span></div>
    <div class="ts-item"><span class="ts-key">Method.</span><span class="ts-text">TODO: 1-line.</span></div>
    <div class="ts-item"><span class="ts-key">Result.</span><span class="ts-text">TODO: 1-line.</span></div>
    <div class="ts-item"><span class="ts-key">Practical.</span><span class="ts-text">TODO: 1-line.</span></div>''',
'''    <div class="ts-title"><span class="num">&#10003;</span> Takeaways</div>
    <div class="ts-item"><span class="ts-key">Idea.</span><span class="ts-text">Estimate the tilt, never the density &mdash; FORS removes discretization bias.</span></div>
    <div class="ts-item"><span class="ts-key">Method.</span><span class="ts-text">Certify the proof numerically: per-step KL by deterministic quadrature, zero MC noise.</span></div>
    <div class="ts-item"><span class="ts-key">Result.</span><span class="ts-text">All 5 claims reproduce; polylog vs poly separation certified to float64 precision.</span></div>
    <div class="ts-item"><span class="ts-key">Cost.</span><span class="ts-text">~10 h CPU + ~$3 GPU; every result rerunnable from the published bundle.</span></div>''')

# --- FOOTER ---
T = T.replace(
'''      <strong class="method-name">METHOD NAME</strong> &middot; Venue Year &middot;
      Acknowledgements: TODO.
    </div>
    <div>
      Code: <span class="repo">github.com/&lt;org&gt;/&lt;repo&gt;</span> &nbsp;&middot;&nbsp;
      Contact: <span class="repo">presenter@example.com</span>
    </div>''',
'''      <strong class="method-name">FORS reproduction</strong> &middot; ICML 2026 agent repro challenge &middot;
      Paper: Chen, Chewi, Daskalakis, Rakhlin (arXiv:2602.01338)
    </div>
    <div>
      Code: <span class="repo">github.com/Auenchanters/ICML-2026</span> &nbsp;&middot;&nbsp;
      Logbook: <span class="repo">Auenchanters/repro-2602-01338-fors</span>
    </div>''')

T = T.replace('<div class="ornament">LAB &middot; INSTITUTION</div>',
              '<div class="ornament">HF &middot; alphaXiv &middot; Trackio</div>')

# title tag
T = re.sub(r'<title>.*?</title>', '<title>FORS Reproduction Poster</title>', T, flags=re.DOTALL)

Path("poster/poster.html").write_text(T, encoding="utf-8")
print("poster.html written:", len(T), "chars")
print("remaining TODOs:", T.count("TODO"))
