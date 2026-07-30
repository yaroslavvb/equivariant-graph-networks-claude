// Paste-into-the-console render check for the tutorial.
//
// tools/checkmods.mjs only *imports* each chapter, which catches syntax and
// export-shape problems but not a render() that throws — and render() is where
// the interesting code lives. This walks every chapter, renders it, and reports
// what came out.
//
// Usage: serve the repo root, open tutorial/index.html, paste this into the
// browser console, and await the promise.

(async () => {
  // Geometry checks are meaningless when the pane is hidden: innerWidth reads 0,
  // every box measures 0x0, and a perfectly good figure looks broken.
  if (!window.innerWidth) {
    console.warn('viewport reports zero width — the browser pane is hidden. ' +
      'Geometry checks would all false-positive; show the pane and re-run.');
    return { skipped: true };
  }
  const ids = ['question', 'problem', 'symmetry', 'blindness', 'machinery', 'couplings',
               'layer', 'lmax', 'forces', 'locality', 'hull', 'genealogy'];
  const rows = [];
  for (const id of ids) {
    location.hash = '#' + id;
    // chapters render on first visit; give async data loads time to settle
    await new Promise((r) => setTimeout(r, 1400));
    const sec = document.getElementById('chapter-' + id);
    if (!sec) { rows.push({ id, ok: false, why: 'section missing' }); continue; }
    const err = sec.querySelector('.note.dat code');
    if (err) { rows.push({ id, ok: false, why: 'render threw: ' + err.textContent }); continue; }
    const checks = [...sec.querySelectorAll('.checkline')];
    const failing = checks.filter((c) => c.querySelector('.mark.bad')).length;

    // A plot fed a bad index produces NaN path coordinates, and because the axis
    // limits are computed across every series, ONE bad series blanks the whole
    // figure — including the correct curves next to it. That failure is silent:
    // the chapter renders, the SVG exists, and nothing throws. Look for it.
    let nanPath = 0, emptyPath = 0, nanText = 0;
    sec.querySelectorAll('svg path').forEach((p) => {
      const d = p.getAttribute('d') || '';
      if (/NaN|Infinity/.test(d)) nanPath++;
      if (d.trim() === '') emptyPath++;
    });
    sec.querySelectorAll('svg text').forEach((t) => {
      if (/NaN|Infinity/.test(t.textContent)) nanText++;
    });
    // Markup passed to h() as a text child instead of via its html: attribute
    // shows up to the reader as literal tags.
    const markupLeak = (sec.innerText.match(/<\/?(em|span|strong|code)[ >]/g) || []).length;

    rows.push({
      id,
      ok: nanPath === 0 && nanText === 0 && markupLeak === 0,
      h2: sec.querySelectorAll('h2').length,
      svg: sec.querySelectorAll('.demo svg').length,
      canvas: sec.querySelectorAll('.demo canvas').length,
      tables: sec.querySelectorAll('table').length,
      controls: sec.querySelectorAll('button, input[type=range]').length,
      checks: checks.length,
      failingChecks: failing,
      nanPath,
      emptyPath,
      nanText,
      markupLeak,
      // a chapter that rendered but produced almost nothing is a silent failure
      thin: sec.innerText.length < 1200,
    });
  }
  const broken = rows.filter((r) => !r.ok);
  const thin = rows.filter((r) => r.ok && r.thin);
  const withFailingChecks = rows.filter((r) => r.ok && r.failingChecks > 0);
  console.table(rows);
  console.log(
    broken.length === 0 && thin.length === 0
      ? `ALL ${rows.length} CHAPTERS RENDER CLEAN`
      : `PROBLEMS — broken: ${broken.map((r) => r.id).join(', ') || 'none'}; ` +
        `suspiciously thin: ${thin.map((r) => r.id).join(', ') || 'none'}`);
  if (withFailingChecks.length) {
    console.log('chapters displaying a failing check (may be intentional — read the prose): ' +
      withFailingChecks.map((r) => `${r.id}(${r.failingChecks})`).join(', '));
  }
  return { rows, broken, thin };
})();
