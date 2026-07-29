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
  const ids = ['question', 'symmetry', 'blindness', 'machinery', 'layer',
               'lmax', 'forces', 'locality', 'hull', 'genealogy'];
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
    rows.push({
      id,
      ok: true,
      h2: sec.querySelectorAll('h2').length,
      svg: sec.querySelectorAll('.demo svg').length,
      canvas: sec.querySelectorAll('.demo canvas').length,
      tables: sec.querySelectorAll('table').length,
      controls: sec.querySelectorAll('button, input[type=range]').length,
      checks: checks.length,
      failingChecks: failing,
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
      ? `ALL ${rows.length} CHAPTERS RENDER`
      : `PROBLEMS — broken: ${broken.map((r) => r.id).join(', ') || 'none'}; ` +
        `suspiciously thin: ${thin.map((r) => r.id).join(', ') || 'none'}`);
  if (withFailingChecks.length) {
    console.log('chapters displaying a failing check (may be intentional — read the prose): ' +
      withFailingChecks.map((r) => `${r.id}(${r.failingChecks})`).join(', '));
  }
  return { rows, broken, thin };
})();
