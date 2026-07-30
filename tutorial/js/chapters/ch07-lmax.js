import { h, segmented, checkLine, loadResults, Plot, PALETTE } from '../ui.js';

// The only numbers on this page that are NOT produced by
// python/experiments/lmax_learning_curves.py. Source: NequIP (Batzner et al.,
// Nat. Commun. 13, 2453, 2022), Table II -- revised-MD17 aspirin force MAE in
// meV/A, a single training budget with no capacity control. Every ratio quoted
// from them below is computed from these entries at render time.
const PAPER = {
  lmax: { 0: 41.9, 1: 12.9, 2: 8.7, 3: 8.5 },
  ace: 17.9,
  gemnet: 9.5,
};

const LCOLOR = { 0: PALETTE[0], 1: PALETTE[1], 2: PALETTE[2], 3: PALETTE[3] };
const pad = (s, n) => String(s).padStart(n);
const last = (a) => a[a.length - 1];

/** Least-squares fit of log10(rmse) on log10(N), matching loglog_fit() in the
 *  Python script exactly, including its r^2 convention. */
function loglogFit(ns, ys) {
  const n = ns.length;
  const x = ns.map((v) => Math.log10(v));
  const y = ys.map((v) => Math.log10(v));
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const mx = mean(x), my = mean(y);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (x[i] - mx) * (y[i] - my); den += (x[i] - mx) ** 2; }
  const slope = num / den;
  const intercept = my - slope * mx;
  const resid = x.map((xi, i) => y[i] - (slope * xi + intercept));
  const varOf = (a) => { const m = mean(a); return mean(a.map((v) => (v - m) ** 2)); };
  return { slope, intercept, fit_r2: 1 - varOf(resid) / varOf(y) };
}

/** Independent on/off buttons, styled like the shared segmented control. */
function multiToggle({ label, options, state, onChange }) {
  const seg = h('span', { class: 'seg' });
  for (const o of options) {
    const b = h('button', {
      type: 'button',
      'aria-pressed': String(!!state[o.value]),
      onclick: () => {
        state[o.value] = !state[o.value];
        b.setAttribute('aria-pressed', String(state[o.value]));
        onChange();
      },
    }, o.label);
    seg.appendChild(b);
  }
  return h('div', { class: 'ctl' }, label ? h('label', {}, label) : null, seg);
}

export default {
  id: 'lmax',
  title: 'The ℓ-max ablation',
  async render(root) {
    const D = await loadResults('lmax_learning_curves');
    const M = D.meta;
    const SP = D.spectra;
    const LC = D.learning_curves;
    const SL = D.slopes;
    const G = D.gains;
    const CM = D.control_matched_features;
    const CR = D.control_radial_capacity;
    const GRID = M.n_train_grid;
    const NMAX = last(GRID);
    const LMAXES = M.lmax_grid;

    // ---- quantities derived live from the loaded results --------------------
    const paperGain = PAPER.lmax[0] / PAPER.lmax[2];
    const gemnetBeatsL1 = PAPER.lmax[1] / PAPER.gemnet;

    const rawFactor = (t) => last(LC[t]['0'].rmse_mean) / last(LC[t]['2'].rmse_mean);
    const matchedFactor = (t) =>
      last(CM.curves[t]['0'].rmse_mean) / last(CM.curves[t]['2'].rmse_mean);
    const retainedPct = (t) => 100 * Math.log(matchedFactor(t)) / Math.log(rawFactor(t));

    /** Smallest training-set size at which l_max = l already beats what
     *  l_max = 0 achieves with the entire N = NMAX budget. */
    const crossoverN = (t, l) => {
      const goal = last(LC[t]['0'].rmse_mean);
      const cur = LC[t][String(l)].rmse_mean;
      for (let i = 0; i < GRID.length; i++) if (cur[i] <= goal) return GRID[i];
      return null;
    };
    const swCross = crossoverN('sw', 2);
    const repCross = crossoverN('rep', 2);
    const noiseRatio2 = LC.sw['2'].rmse_over_noise_scale_at_max_n;
    const withinPct = Math.abs(100 * (noiseRatio2 - 1));

    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 7'),
      h('h1', {}, 'The ℓ-max ablation'),
      h('p', { class: 'lede' },
        'This is NequIP’s central causal experiment: raise the maximum tensor rank of the ' +
        'internal features, hold everything else fixed, and watch the error fall. The chapter ' +
        'rebuilds it at a scale where the answer is decidable rather than merely observed, adds ' +
        'the two fairness controls the headline table does not have, and then runs into a target ' +
        'for which the paper’s qualitative pattern is simply false. That failure turns out to be ' +
        'the most useful thing here.'),

      h('h2', {}, 'The claim, and two things wrong with how it is usually quoted'),
      h('p', { class: 'prose', html:
        'The quotable version is a single row of NequIP’s Table II: on revised-MD17 aspirin the ' +
        `force MAE falls ${PAPER.lmax[0]} → ${PAPER.lmax[1]} → ${PAPER.lmax[2]} → ` +
        `${PAPER.lmax[3]} meV/Å as $\\ell_{\\max}$ goes 0, 1, 2, 3 — a factor of ` +
        `${paperGain.toFixed(1)} from the scalar model to $\\ell_{\\max} = 2$, with the largest ` +
        'single step at the very beginning and clear diminishing returns after. It is a clean, ' +
        'memorable curve, and it is the reason “higher $\\ell$” became a design axis. Two ' +
        'qualifications have to travel with it, and they usually do not.' }),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'Caveat 1 — which experiment is actually capacity-controlled'),
        h('div', { html:
          'The $\\ell = 0..3$ scan in Table II is on rMD17, but it is a <em>single-budget ' +
          'accuracy comparison with no capacity control</em>: higher $\\ell$ also means more ' +
          'features and more weights. The genuinely capacity-controlled ablation — the weight- ' +
          'and feature-matched $\\ell = 0$ baselines — and the data-efficiency learning curves ' +
          'live in Appendix B, Fig. 11, and they are on the <strong>original MD-17</strong>, not ' +
          'rMD17. Quoting the rMD17 numbers as though they were the controlled result conflates ' +
          'two different experiments on two different datasets.' })),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'Caveat 2 — the ℓ = 0 baseline is weak for two reasons at once'),
        h('div', { html:
          `In the same table, ACE — invariant and linear — scores ${PAPER.ace} meV/Å, and ` +
          `GemNet-T/Q — invariant but <em>directional</em> — scores ${PAPER.gemnet}, which is ` +
          `${gemnetBeatsL1.toFixed(2)}× better than NequIP at $\\ell_{\\max} = 1$ ` +
          `(${PAPER.lmax[1]}). An invariant model therefore beats an equivariant one here. So ` +
          `the ${PAPER.lmax[0]} meV/Å of NequIP $\\ell = 0$ is not a measurement of “what you ` +
          'lose without equivariance”; it is largely a measurement of what you lose without ' +
          '<em>angular information of any kind</em>. The same pattern shows up in Table I on ' +
          'original MD-17, where SchNet (distances only) sits at 58.5 against DimeNet ' +
          '(invariant plus angles) at 21.6. This is the main confound in the headline ablation. ' +
          'The experiment below is built so that it can be measured rather than argued about: ' +
          'every model here is an invariant readout of an equivariant layer, and the only thing ' +
          'that changes between them is the angular rank retained.' })),

      h('h2', {}, 'A system in which the question is decidable'),
      h('p', { class: 'prose', html:
        `One central atom, ${M.n_neighbours} neighbours drawn uniformly by volume in the shell ` +
        `$${M.shell[0]} \\le r \\le ${M.shell[1]}$, rejecting any configuration with two ` +
        `neighbours closer than ${M.min_neighbour_separation} (acceptance rate ` +
        `${(100 * M.acceptance_rate).toFixed(1)}%). The energy is analytic: a Morse two-body term ` +
        `($D_e = ${D.system.morse.D_e}$, $r_e = ${D.system.morse.r_e}$, $a = ${D.system.morse.a}$) ` +
        'times a smooth polynomial envelope that vanishes to second order at ' +
        `$r_{\\mathrm{cut}} = ${M.r_cut}$, plus a three-body term. Two three-body kernels are ` +
        'studied, and the difference between them is the entire point.' }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'target'), h('th', {}, 'three-body kernel'),
          h('th', { class: 'num' }, 'Var(E₃)/Var(E)'),
          h('th', { class: 'num' }, 'corr(E₂,E₃)'),
          h('th', { class: 'num' }, 'radial-only nRMSE floor'))),
        h('tbody', {},
          ...['sw', 'rep'].map((t) => h('tr', {},
            h('td', {}, t),
            h('td', {}, M.targets[t]),
            h('td', { class: 'num' }, D.system[t].var_frac_three_body.toFixed(3)),
            h('td', { class: 'num' }, D.system[t].corr_two_three.toFixed(3)),
            h('td', { class: 'num' }, D.system[t].angular_blind_nrmse.toFixed(4)))))),

      h('p', { class: 'prose', html:
        'That last column is the floor a model with <em>no angular information at all</em> runs ' +
        `into. Given ${D.system.sw.angular_blind_n_features} purely radial invariants and the ` +
        `whole ${M.n_pool}-configuration pool fitted in-sample, ` +
        `${(100 * D.system.sw.irreducible_angular_var_frac).toFixed(1)}% of the Stillinger–Weber ` +
        `energy variance and ${(100 * D.system.rep.irreducible_angular_var_frac).toFixed(1)}% of ` +
        'the repulsion energy variance remain unreachable. There is real angular structure to ' +
        'find. The covariance between the two-body and three-body parts is strongly negative in ' +
        'both cases, which is why the variance fractions do not sum to one.' }),
      h('p', { class: 'prose', html:
        'The features are exactly the first layer of a NequIP convolution, ' +
        '$f_\\ell[n] = \\sum_j R_n(r_{ij})\\, Y_\\ell(\\hat r_{ij})$, with a Gaussian radial ' +
        'basis multiplied by the same envelope. Because the energy is a scalar, the readout uses ' +
        'the complete set of second-order $O(3)$ invariants of those features: the linear ' +
        '$\\ell = 0$ channels plus every cross-channel contraction ' +
        '$f_\\ell[a]\\cdot f_\\ell[b]$, so ' +
        '$P = n_{\\mathrm{rad}} + (\\ell_{\\max}+1)\\,n_{\\mathrm{rad}}(n_{\\mathrm{rad}}+1)/2$. ' +
        'The columns are ordered so that the $\\ell_{\\max} = L$ feature set is a strict ' +
        '<em>prefix</em> of any larger one; the ablation is a clean nesting rather than four ' +
        'unrelated models. The regressor is closed-form ridge with the penalty chosen per fit by ' +
        `an ${M.alpha_selection}. Training labels carry Gaussian noise at ` +
        `${(100 * M.label_noise_frac_of_std).toFixed(0)}% of the energy standard deviation, test ` +
        `energies are exact, and every point is averaged over ${M.n_resamples} independent ` +
        'resamples of the pool. The constructed invariants are invariant to machine precision ' +
        `(max relative deviation ${D.invariance.feature_max_rel_dev.toExponential(2)} under a ` +
        'random rotation), so nothing below is an artefact of a broken feature map.' }),

      h('h2', {}, 'The angular spectrum of the target is the whole story'),
      h('p', { class: 'prose', html:
        'Before looking at a single learning curve, expand each three-body kernel in Legendre ' +
        'polynomials of the bond angle. The Stillinger–Weber kernel is ' +
        '$\\lambda(\\cos\\theta - \\cos\\theta_0)^2$, and the square of a linear function of ' +
        '$\\cos\\theta$ has <em>exactly three</em> Legendre terms: ' +
        `$a_0 = ${SP.sw_legendre_coeffs[0].toFixed(4)}$, ` +
        `$a_1 = ${SP.sw_legendre_coeffs[1].toFixed(4)}$, ` +
        `$a_2 = ${SP.sw_legendre_coeffs[2].toFixed(4)}$, and identically zero beyond. ` +
        'Numerically the largest coefficient above $\\ell = 2$ is ' +
        `${SP.sw_max_abs_coeff_above_l2.toExponential(2)}, which is quadrature noise. The ` +
        'repulsion kernel $\\mu/|v_j - v_k|$ expands as ' +
        '$\\sum_\\ell (r_<^\\ell / r_>^{\\ell+1}) P_\\ell$, an infinite series: ' +
        `${(100 * SP.rep_power_frac_above_l2).toFixed(2)}% of its angular power sits at ` +
        '$\\ell \\ge 3$ and it never terminates.' }));

    const specDemo = h('div', { class: 'demo' });
    specDemo.append(h('h3', {}, 'Angular power per degree, for the two three-body kernels'),
      h('p', { class: 'hint' },
        'One kernel is a degree-2 polynomial in the bond-angle cosine and therefore stops dead. ' +
        'The other is a Coulomb-like expansion with weight at every degree. A real molecular ' +
        'potential-energy surface looks like the second one.'));
    const specPlot = new Plot({ width: 660, height: 300,
      xLabel: 'degree ℓ', yLabel: 'fraction of total angular power' });
    specPlot.add({ points: SP.l.map((l, i) => [l, SP.sw_power_frac[i]]),
      color: PALETTE[0], width: 2.4, markers: true, r: 4,
      label: 'sw:  λ(cos θ − cos θ₀)²  — exactly zero above ℓ = 2' });
    specPlot.add({ points: SP.l.map((l, i) => [l, SP.rep_power_frac[i]]),
      color: PALETTE[1], width: 2.4, markers: true, r: 4, dash: '5 4',
      label: 'rep:  μ / |vⱼ − v_k|  — a tail at every degree' });
    specPlot.setLimits([-0.25, 6.25], [-0.03, 0.72]);
    specDemo.append(specPlot.render(), specPlot.legend(),
      h('div', { class: 'readout' },
        'angular power captured by truncating at l_max (cumulative fraction):\n' +
        '   l_max          sw         rep\n' +
        SP.l.map((l, i) => `   ${pad(l, 2)}     ` +
          `${pad(SP.sw_power_frac.slice(0, i + 1).reduce((a, b) => a + b, 0).toFixed(6), 11)}` +
          `  ${pad(SP.rep_cumulative_power_frac[i].toFixed(6), 10)}`).join('\n')));
    root.append(specDemo);

    root.append(
      h('p', { class: 'prose', html:
        'Two numbers from that figure are worth holding on to. Of the <em>non-constant</em> ' +
        'angular power in the Stillinger–Weber kernel, $\\ell = 1$ accounts for ' +
        `${(100 * SP.sw_power_frac_nonconstant[0]).toFixed(1)}% and $\\ell = 2$ for ` +
        `${(100 * SP.sw_power_frac_nonconstant[1]).toFixed(1)}%. Measured in power removed, the ` +
        'first step is genuinely the bigger one, exactly as the paper’s intuition would have it. ' +
        'Measured in error <em>ratio</em>, as we are about to see, it is not even close.' }),

      h('h2', {}, 'The learning curves'),
      h('p', { class: 'prose', html:
        'Everything below comes from one plot. The default view is the main experiment on the ' +
        `Stillinger–Weber target: $n_{\\mathrm{rad}} = ${M.n_rad_main}$ held fixed, so the ` +
        'feature count grows with $\\ell_{\\max}$ exactly as it does in the paper’s Table II.' }));

    // ------------------------------------------------------------------
    // Interactive learning-curve plot
    // ------------------------------------------------------------------
    const state = { target: 'sw', variant: 'main', fits: true, noise: false,
      show: { 0: true, 1: true, 2: true, 3: true } };
    const curveOf = (t, l) =>
      (state.variant === 'main' ? LC[t][String(l)] : CM.curves[t][String(l)]);
    const fitWindow = GRID.map((n, i) => [n, i]).filter(([n]) => M.slope_fit_points.includes(n));
    const fitOf = (c) => loglogFit(fitWindow.map(([n]) => n),
      fitWindow.map(([, i]) => c.rmse_mean[i]));

    const lcDemo = h('div', { class: 'demo' });
    lcDemo.append(h('h3', {}, 'Test RMSE against training-set size, log–log'),
      h('p', { class: 'hint' },
        'Switch target to run the same ablation against a kernel whose angular spectrum never ' +
        'terminates. Switch to the matched-P control to hold the feature count nearly fixed ' +
        'across ℓ_max — that is the difference between a demonstration and a confound. Dashed ' +
        'straight lines are the fitted power laws; the faint dotted lines are the noise-limited ' +
        'scale σ√(P/N).'));

    const plotHolder = h('div');
    const legendHolder = h('div');
    const readHolder = h('div', { class: 'readout' });

    const redraw = () => {
      const t = state.target;
      const vis = LMAXES.filter((l) => state.show[l]);
      const plot = new Plot({ width: 700, height: 400, xLog: true, yLog: true,
        xLabel: 'training-set size N', yLabel: 'test RMSE (energy units)' });

      const ys = [];
      for (const l of vis) for (const v of curveOf(t, l).rmse_mean) ys.push(v);
      if (!ys.length) ys.push(1e-2, 1);
      plot.setLimits([GRID[0] / 1.6, NMAX * 1.6],
        [Math.min(...ys) / 2.2, Math.max(...ys) * 2.2]);

      const sigma = D.system[t].label_noise_sigma;
      for (const l of vis) {
        const c = curveOf(t, l);
        if (state.noise) {
          plot.add({ points: GRID.map((n) => [n, sigma * Math.sqrt(c.n_features / n)]),
            color: LCOLOR[l], width: 1, dash: '1 4', opacity: 0.8 });
        }
        if (state.fits) {
          const f = fitOf(c);
          plot.add({
            points: [fitWindow[0][0], last(fitWindow)[0]].map(
              (n) => [n, Math.pow(10, f.intercept + f.slope * Math.log10(n))]),
            color: LCOLOR[l], width: 1.4, dash: '7 4', opacity: 0.9 });
        }
        plot.add({ points: GRID.map((n, i) => [n, c.rmse_mean[i]]),
          color: LCOLOR[l], width: 2.4, markers: true, r: 3.4,
          label: `ℓ_max = ${l}   (P = ${c.n_features})` });
      }

      plotHolder.innerHTML = '';
      plotHolder.appendChild(plot.render());
      legendHolder.innerHTML = '';
      legendHolder.appendChild(plot.legend());

      const lines = [];
      lines.push(`target '${t}',  ` + (state.variant === 'main'
        ? `n_rad = ${M.n_rad_main} fixed, so P grows with l_max`
        : 'n_rad tuned per l_max to hold P nearly fixed'));
      lines.push('');
      lines.push(`  l_max     P    RMSE(N=${NMAX})     slope   fit R^2       floor`);
      for (const l of LMAXES) {
        if (!state.show[l]) continue;
        const c = curveOf(t, l);
        const f = fitOf(c);
        lines.push(`  ${pad(l, 4)}  ${pad(c.n_features, 5)}` +
          `  ${pad(last(c.rmse_mean).toFixed(5), 12)}` +
          `  ${pad(f.slope.toFixed(4), 9)}  ${pad(f.fit_r2.toFixed(4), 7)}` +
          `  ${pad(c.floor_rmse.toExponential(2), 10)}`);
      }
      if (state.variant === 'main') {
        let worst = 0;
        for (const l of LMAXES) {
          worst = Math.max(worst, Math.abs(fitOf(LC[t][String(l)]).slope - SL[t][String(l)].slope));
        }
        lines.push('');
        lines.push('  slopes refitted in the browser agree with the stored Python fit to ' +
          `${worst.toExponential(1)}`);
      }
      lines.push('');
      lines.push(`  fit window N = ${M.slope_fit_points.join(', ')};  ` +
        `label noise sigma = ${sigma.toFixed(5)};  ${M.n_resamples} resamples per point`);
      readHolder.textContent = lines.join('\n');
    };

    lcDemo.append(plotHolder, legendHolder,
      h('div', { class: 'controls' },
        segmented({ label: 'target',
          options: [{ label: 'sw — terminates at ℓ=2', value: 'sw' },
            { label: 'rep — infinite spectrum', value: 'rep' }],
          value: 'sw', onPick: (v) => { state.target = v; redraw(); } }),
        segmented({ label: 'experiment',
          options: [{ label: 'main', value: 'main' },
            { label: 'matched-P control', value: 'ctl' }],
          value: 'main', onPick: (v) => { state.variant = v; redraw(); } })),
      h('div', { class: 'controls' },
        multiToggle({ label: 'curves',
          options: LMAXES.map((l) => ({ label: `ℓ ≤ ${l}`, value: l })),
          state: state.show, onChange: redraw }),
        multiToggle({ label: 'overlays',
          options: [{ label: 'power-law fits', value: 'fits' },
            { label: 'σ√(P/N)', value: 'noise' }],
          state, onChange: redraw })),
      readHolder);
    root.append(lcDemo);
    redraw();

    root.append(
      h('p', { class: 'prose', html:
        `The $\\ell_{\\max} = 0$ curve is almost flat: its fitted slope is ` +
        `${SL.sw['0'].slope.toFixed(4)}, meaning that across the fit window ` +
        `$N = ${M.slope_fit_points[0]}..${NMAX}$ it barely improves at all. More training data ` +
        'does not help a model that cannot represent the function. At the other extreme, ' +
        `$\\ell_{\\max} = 2$ falls from ${LC.sw['2'].rmse_mean[0].toFixed(5)} at ` +
        `$N = ${GRID[0]}$ to ${last(LC.sw['2'].rmse_mean).toFixed(5)} at $N = ${NMAX}$ and is ` +
        'still descending steeply at the right edge. The two curves are not offset copies of one ' +
        'another; they have different <em>shapes</em>, and that is the substance of the ' +
        'data-efficiency half of the claim.' }),
      h('p', { class: 'prose', html:
        `As a data-efficiency statement: with ${swCross} training configurations the ` +
        `$\\ell_{\\max} = 2$ model is already better than what $\\ell_{\\max} = 0$ manages with ` +
        `all ${NMAX}, a factor of ${(NMAX / swCross).toFixed(0)}× in data — and “better” ` +
        'understates it, because the scalar model would not reach that error with any amount of ' +
        `data at all. On the repulsion target the crossover is at ${repCross}, again ` +
        `${(NMAX / repCross).toFixed(0)}×.` }),

      h('h2', {}, 'Fairness control A: the same number of features for everyone'),
      h('p', { class: 'prose', html:
        'The obvious objection to that plot is that raising $\\ell_{\\max}$ at fixed ' +
        `$n_{\\mathrm{rad}} = ${M.n_rad_main}$ also raises the feature count, from ` +
        `$P = ${LC.sw['0'].n_features}$ to $P = ${LC.sw['3'].n_features}$, so part of any gain ` +
        'could be raw capacity. This is precisely the objection that NequIP’s Table II does not ' +
        'answer. The control tunes $n_{\\mathrm{rad}}$ per $\\ell_{\\max}$ so that all four ' +
        'models get nearly the same number of invariants, and it deliberately hands the scalar ' +
        'model the <em>largest</em> budget of the four.' }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'ℓ_max'), h('th', { class: 'num' }, 'n_rad'),
          h('th', { class: 'num' }, 'P'),
          h('th', { class: 'num' }, `sw RMSE at N=${NMAX}`),
          h('th', { class: 'num' }, `rep RMSE at N=${NMAX}`))),
        h('tbody', {},
          ...LMAXES.map((l) => h('tr', l === 2 ? { class: 'hi' } : {},
            h('td', {}, String(l)),
            h('td', { class: 'num' }, String(CM.n_rad_per_lmax[String(l)])),
            h('td', { class: 'num' }, String(CM.n_features[String(l)])),
            h('td', { class: 'num' }, last(CM.curves.sw[String(l)].rmse_mean).toFixed(5)),
            h('td', { class: 'num' }, last(CM.curves.rep[String(l)].rmse_mean).toFixed(5)))))),
      h('p', { class: 'prose', html:
        'Essentially none of the gain was capacity. On the Stillinger–Weber target the raw ' +
        `$\\ell = 0 \\to 2$ improvement at $N = ${NMAX}$ is a factor of ` +
        `${rawFactor('sw').toFixed(1)}; under the matched-$P$ control it is ` +
        `${matchedFactor('sw').toFixed(1)}, which is <em>larger</em> — ` +
        `${retainedPct('sw').toFixed(0)}% of the log-gain retained. On the repulsion target it is ` +
        `${rawFactor('rep').toFixed(3)} raw against ${matchedFactor('rep').toFixed(3)} matched, ` +
        `${retainedPct('rep').toFixed(0)}% retained. Handing the scalar model more parameters ` +
        'makes it marginally worse, because the extra radial channels are estimation variance ' +
        'spent on a function it still cannot represent. That is the difference between a ' +
        'demonstration and a confound, and it is the measurement that turns the observation into ' +
        'an argument.' }),
      h('p', { class: 'prose', html:
        'One artefact worth naming rather than hiding. In the matched-$P$ view at ' +
        `$N = ${GRID[0]}$ the $\\ell_{\\max} = 0$ model has ${CM.n_features['0']} features and ` +
        `${GRID[0]} training points, and its test RMSE is ` +
        `${CM.curves.sw['0'].rmse_mean[0].toFixed(2)} — about ` +
        `${(CM.curves.sw['0'].rmse_mean[0] / D.system.sw.std_energy).toFixed(0)}× the standard ` +
        'deviation of the energy itself. Ridge with a validated penalty does not save you when ' +
        '$P \\gg N$ and the validation split is two points wide. The left edge of that curve is ' +
        `noise, which is why the slope fits start at $N = ${M.slope_fit_points[0]}$.` }),

      h('h2', {}, 'Fairness control B: radial resolution is not a substitute for angular rank'),
      h('p', { class: 'prose', html:
        'The subtler objection is that a scalar model might simply need a finer radial basis. It ' +
        'does not. Growing the $\\ell_{\\max} = 0$ feature count by a factor of ' +
        `${(last(CR.targets.sw.lmax0.n_features) / CR.targets.sw.lmax0.n_features[0]).toFixed(0)} ` +
        `and fitting on the entire ${M.n_pool}-configuration pool moves the error by essentially ` +
        'nothing.' }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'model'), h('th', { class: 'num' }, 'n_rad'),
          h('th', { class: 'num' }, 'P'),
          h('th', { class: 'num' }, 'sw test RMSE'),
          h('th', { class: 'num' }, 'rep test RMSE'))),
        h('tbody', {},
          ...CR.n_rad_grid.map((nr, i) => h('tr', {},
            h('td', {}, 'ℓ_max = 0'),
            h('td', { class: 'num' }, String(nr)),
            h('td', { class: 'num' }, String(CR.targets.sw.lmax0.n_features[i])),
            h('td', { class: 'num' }, CR.targets.sw.lmax0.rmse[i].toFixed(5)),
            h('td', { class: 'num' }, CR.targets.rep.lmax0.rmse[i].toFixed(5)))),
          ...[1, 2, 3].map((l) => h('tr', l === 2 ? { class: 'hi' } : {},
            h('td', {}, `ℓ_max = ${l}`),
            h('td', { class: 'num' }, String(CR.targets.sw.small_lmax_reference[String(l)].n_rad)),
            h('td', { class: 'num' },
              String(CR.targets.sw.small_lmax_reference[String(l)].n_features)),
            h('td', { class: 'num' },
              CR.targets.sw.small_lmax_reference[String(l)].rmse.toFixed(5)),
            h('td', { class: 'num' },
              CR.targets.rep.small_lmax_reference[String(l)].rmse.toFixed(5)))))),
      h('p', { class: 'prose', html:
        `The scalar model plateaus: from ${CR.targets.sw.lmax0.n_features[0]} to ` +
        `${last(CR.targets.sw.lmax0.n_features)} radial invariants the Stillinger–Weber error ` +
        `goes ${CR.targets.sw.lmax0.rmse[0].toFixed(5)} → ` +
        `${last(CR.targets.sw.lmax0.rmse).toFixed(5)}, that is, ` +
        `${(100 * (last(CR.targets.sw.lmax0.rmse) / CR.targets.sw.lmax0.rmse[0] - 1)).toFixed(1)}% ` +
        `<em>worse</em>. Meanwhile $\\ell_{\\max} = 2$ with ` +
        `${CR.targets.sw.small_lmax_reference['2'].n_features} features beats it by a factor of ` +
        `${(last(CR.targets.sw.lmax0.rmse) / CR.targets.sw.small_lmax_reference['2'].rmse).toFixed(1)} ` +
        `using ${(last(CR.targets.sw.lmax0.n_features) / CR.targets.sw.small_lmax_reference['2'].n_features).toFixed(1)}× ` +
        `fewer of them; on the repulsion target the same comparison gives a factor of ` +
        `${(last(CR.targets.rep.lmax0.rmse) / CR.targets.rep.small_lmax_reference['2'].rmse).toFixed(1)}. ` +
        'The limitation is representational, not parametric. Note that this is the controlled ' +
        'version of Caveat 2: the weak model here is not merely non-equivariant, it is angularly ' +
        'blind, and these two tables measure exactly how much of the gap that accounts for.' }),

      h('h2', {}, 'The honest negative: the paper’s ordering does not survive'),
      h('p', { class: 'prose', html:
        'Here is where the toy refuses to behave. NequIP’s pattern is that the $0 \\to 1$ step is ' +
        'the largest and the gains diminish from there. On the Stillinger–Weber target that is ' +
        'flatly false, and not by a little.' }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'step'),
          h('th', { class: 'num' }, `sw: RMSE ratio at N=${NMAX}`),
          h('th', { class: 'num' }, 'sw: floor ratio'),
          h('th', { class: 'num' }, `rep: RMSE ratio at N=${NMAX}`),
          h('th', { class: 'num' }, 'rep: floor ratio'))),
        h('tbody', {},
          ...[['0 → 1', 'gain_0_to_1', 'floor_ratio_0_to_1'],
            ['1 → 2', 'gain_1_to_2', 'floor_ratio_1_to_2'],
            ['2 → 3', 'gain_2_to_3', 'floor_ratio_2_to_3']].map(([lab, gk, fk], i) =>
            h('tr', i === 1 ? { class: 'hi' } : {},
              h('td', {}, lab),
              h('td', { class: 'num' }, `${G.sw[gk].toFixed(2)}×`),
              h('td', { class: 'num' }, `${G.sw[fk].toFixed(2)}×`),
              h('td', { class: 'num' }, `${G.rep[gk].toFixed(2)}×`),
              h('td', { class: 'num' }, `${G.rep[fk].toFixed(2)}×`))))),
      h('p', { class: 'prose', html:
        `The Stillinger–Weber column reads ${G.sw.gain_0_to_1.toFixed(2)}×, then ` +
        `${G.sw.gain_1_to_2.toFixed(1)}×, then ${G.sw.gain_2_to_3.toFixed(2)}×. The middle step ` +
        `is ${(G.sw.gain_1_to_2 / G.sw.gain_0_to_1).toFixed(0)}× larger than the first, and the ` +
        'last step is <em>below one</em>, meaning $\\ell_{\\max} = 3$ is worse than ' +
        '$\\ell_{\\max} = 2$. The representational floors say why in the plainest possible ' +
        `terms: the $1 \\to 2$ floor ratio is ${G.sw.floor_ratio_1_to_2.toFixed(0)}×, because at ` +
        '$\\ell = 2$ the model stops approximating the target and starts <em>containing</em> it. ' +
        'The Legendre expansion of $(\\cos\\theta - \\cos\\theta_0)^2$ terminates at degree 2, so ' +
        'the $1 \\to 2$ step is not an improvement in approximation quality at all — it is a ' +
        'discontinuous jump to zero bias, and a ratio taken across it does not mean what the ' +
        'paper’s ratios mean.' }),
      h('p', { class: 'prose', html:
        `The $2 \\to 3$ floor ratio of ${G.sw.floor_ratio_2_to_3.toFixed(2)}× is not a ` +
        'contradiction of that. It says only that $\\ell_{\\max} = 3$ has a slightly lower ' +
        'residual floor, which is radial-truncation noise; both floors already sit factors of ' +
        `${(last(LC.sw['2'].rmse_mean) / LC.sw['2'].floor_rmse).toFixed(0)} and ` +
        `${(last(LC.sw['3'].rmse_mean) / LC.sw['3'].floor_rmse).toFixed(0)} below the errors the ` +
        `two models actually reach at $N = ${NMAX}$, so the extra headroom is unreachable and ` +
        'the extra features cost more in estimation variance than they return.' }),
      h('p', { class: 'prose', html:
        'Now switch the target in the plot above to <code>rep</code>. The repulsion kernel’s ' +
        'spectrum never terminates, which is the situation of a real molecular potential-energy ' +
        'surface, and the paper’s ordering comes straight back: ' +
        `${G.rep.gain_0_to_1.toFixed(2)}×, ${G.rep.gain_1_to_2.toFixed(2)}×, ` +
        `${G.rep.gain_2_to_3.toFixed(2)}× — monotonically diminishing, exactly as reported. ` +
        'Nothing was tuned to produce either outcome; the two targets differ only in the choice ' +
        'of three-body kernel, and both results are in the ledger.' }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'What the failure actually teaches'),
        h('div', { html:
          'The shape of the $\\ell_{\\max}$ curve is a property of the <strong>target’s angular ' +
          'complexity</strong>, not a universal law about equivariant networks. Where the ' +
          'target’s angular spectrum terminates, the curve has a cliff at the completing degree ' +
          'and is flat on either side of it. Where the spectrum does not terminate, you get ' +
          'smooth diminishing returns. NequIP’s aspirin curve is the second kind; reading it as ' +
          'a general law — “$\\ell = 1$ buys the most and then it tails off” — predicts the ' +
          'wrong thing for the first kind. The corollary the paper does see also shows up here: ' +
          'past the rank actually present in the target, extra $\\ell$ costs accuracy at fixed ' +
          `$N$, because the additional features are pure estimation variance. That is the ` +
          `${G.sw.gain_2_to_3.toFixed(3)}× entry above — a measurable penalty for asking for ` +
          'angular resolution the target does not have.' })),
      h('p', { class: 'prose', html:
        'Note what the angular-power figure said earlier. In <em>power</em> terms $\\ell = 1$ ' +
        'does remove more of the Stillinger–Weber kernel than $\\ell = 2$ ' +
        `(${(100 * SP.sw_power_frac_nonconstant[0]).toFixed(1)}% against ` +
        `${(100 * SP.sw_power_frac_nonconstant[1]).toFixed(1)}%), which is the paper’s intuition, ` +
        'and it is correct. The error ratio nevertheless inverts the ordering, because a ratio ' +
        'is dominated by whichever step reaches zero bias. Two reasonable ways of measuring the ' +
        'same thing disagree about which step matters most, and only one of them is the one ' +
        'everybody quotes.' }),

      h('h2', {}, 'What the slopes say, and one prediction that did not hold'),
      h('p', { class: 'prose', html:
        'Turn on the σ√(P/N) overlay in the plot. That dotted line is where an unbiased ' +
        'least-squares fit would sit given the label noise and the feature count: the error you ' +
        'cannot avoid from finite data alone. A curve far above it is bias-limited, held up by ' +
        'what the model cannot represent. A curve lying on it is noise-limited, and the only ' +
        'cure is more data.' }),
      h('div', { class: 'readout' },
        `target 'sw', main experiment, at N = ${NMAX}:\n\n` +
        LMAXES.map((l) => `  l_max = ${l}   ` +
          `slope ${pad(SL.sw[String(l)].slope.toFixed(4), 8)}   ` +
          `RMSE / sigma sqrt(P/N) = ` +
          `${pad(LC.sw[String(l)].rmse_over_noise_scale_at_max_n.toFixed(2), 6)}`).join('\n')),
      h('p', { class: 'prose', html:
        'The two extremes are unambiguous. $\\ell_{\\max} = 0$ sits ' +
        `${LC.sw['0'].rmse_over_noise_scale_at_max_n.toFixed(1)}× above the noise scale with ` +
        `slope ${SL.sw['0'].slope.toFixed(4)}: bias-limited, and no amount of data will move it. ` +
        `$\\ell_{\\max} = 2$ sits at ${noiseRatio2.toFixed(2)}× the noise scale, within ` +
        `${withinPct.toFixed(0)}% of it, and is still falling. Raising $\\ell_{\\max}$ does not ` +
        'merely shift the curve down, it steepens it: slopes ' +
        `${SL.sw['0'].slope.toFixed(3)}, ${SL.sw['1'].slope.toFixed(3)}, ` +
        `${SL.sw['2'].slope.toFixed(3)} for $\\ell_{\\max} = 0, 1, 2$. That is the ` +
        'data-efficiency claim, reproduced.' }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'A prediction written down before running, which did not hold'),
        h('div', { html:
          'The prediction was that the complete model would be noise-limited in the textbook ' +
          'sense, with a log–log slope of $-1/2$. It is not. The measured slope over ' +
          `$N = ${M.slope_fit_points[0]}..${NMAX}$ is ${SL.sw['2'].slope.toFixed(4)} ` +
          `(fit $R^2 = ${SL.sw['2'].fit_r2.toFixed(4)}$), a factor ` +
          `${(SL.sw['2'].slope / -0.5).toFixed(2)} steeper than $-1/2$, and a check asserting ` +
          '$-1/2$ would have failed. The check in the script now reports the measured value ' +
          'rather than asserting the predicted one, and says so in its own name — which is why ' +
          'the ledger below reads all-pass while still recording a prediction that was wrong. ' +
          'The diagnosis is in the data: the validated ridge penalty is itself still shrinking ' +
          'across the fit window, with median $\\alpha$ falling ' +
          `${LC.sw['2'].alpha_median[3].toExponential(1)} → ` +
          `${last(LC.sw['2'].alpha_median).toExponential(1)}, because the power-spectrum ` +
          'features are strongly collinear. Shrinkage bias is decaying on top of the ' +
          '$\\sqrt{P/N}$ variance and the two effects compound. What pins the “noise-limited” ' +
          `reading is the magnitude rather than the slope: at $N = ${NMAX}$ the error matches ` +
          `$\\sigma\\sqrt{P/N}$ to within ${withinPct.toFixed(0)}%.` })),
      h('p', { class: 'prose', html:
        'A last piece of honesty about the word “complete”. At $\\ell_{\\max} = 2$ the model is ' +
        'complete for the <em>angular</em> part of the Stillinger–Weber target but not literally ' +
        'exact, because the radial basis is finite. The noiseless in-sample floor is nRMSE ' +
        `${LC.sw['2'].floor_nrmse.toExponential(2)}, and the envelope the true energy uses is ` +
        `represented to ${D.radial_basis.gate_rel_l2_error_n8.toExponential(2)} relative $L^2$ ` +
        `error by the ${M.n_rad_main} radial basis functions in play, against ` +
        `${D.radial_basis.gate_rel_l2_error_n4.toExponential(2)} with only four. The residual ` +
        'misspecification at the complete order is radial truncation and nothing else.' }));

    const ledger = h('div', { class: 'demo' });
    const nFail = D.checks.filter((c) => !c.passed).length;
    ledger.append(h('h3', {}, 'Check ledger — python/experiments/lmax_learning_curves.py'),
      h('p', { class: 'hint' },
        'Reproduce with:  uv run python python/experiments/lmax_learning_curves.py'));
    const inner = h('div');
    for (const c of D.checks) inner.append(checkLine(c.passed, c.name, c.detail));
    inner.append(h('div', { class: 'checkline', style: { marginTop: '10px', fontWeight: '700' } },
      h('span', { class: `mark ${nFail === 0 ? 'ok' : 'bad'}` },
        `${D.checks.length - nFail}/${D.checks.length}`),
      h('span', {}, 'PASS')));
    ledger.append(inner);
    root.append(ledger);

    root.append(
      h('p', { class: 'prose', html:
        `Read that ledger with two things in mind. It is ${D.checks.length - nFail} of ` +
        `${D.checks.length}, but two of those passes are passes on <em>negative</em> statements. ` +
        'One check is named “the paper’s pattern FAILS” and asserts precisely that the ' +
        'Stillinger–Weber ordering inverts; a green mark there means the toy reliably fails to ' +
        'reproduce NequIP’s qualitative curve, and that failure is the result. Another reports a ' +
        `measured slope of ${SL.sw['2'].slope.toFixed(4)} where $-1/2$ was predicted, having ` +
        'been rewritten from an assertion into a measurement after the prediction did not ' +
        'survive contact with the data. A ledger that is green because its claims were weakened ' +
        'is worth less than one that is green because its claims were strong, so the weakening ' +
        'is stated here instead of being left to be inferred from the check names.' }),
      h('p', { class: 'prose', html:
        'What survives is the part of NequIP’s argument that is genuinely about representation. ' +
        'Angular rank buys something that radial resolution and extra parameters cannot: under a ' +
        'control that gives the scalar model the most features of anyone, ' +
        `${retainedPct('sw').toFixed(0)}% of the log-gain remains on one target and ` +
        `${retainedPct('rep').toFixed(0)}% on the other, and the learning curves change slope ` +
        'rather than merely shifting down. What does not survive is the tidy shape of the curve. ' +
        'Whether the first step or the third is the big one depends on where the target’s ' +
        'angular spectrum actually lives, so the honest form of the claim is conditional: raise ' +
        '$\\ell_{\\max}$ until you reach the angular complexity of your target, and not past it. ' +
        'Chapter 11 returns to this, because the descriptor tradition solved the same problem by ' +
        'choosing the body order in advance, and several models near the top of today’s ' +
        'leaderboard came down that road instead.' }));
  },
};
