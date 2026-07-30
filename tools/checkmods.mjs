// Parse/shape check for the tutorial's ES modules. A DOM-related runtime error
// means the file parsed, which is all we can verify outside a browser.
import fs from 'fs';
const base = process.argv[2];
const domish = (m) => /document is not defined|window is not defined|fetch failed|is not defined/.test(m);
let bad = 0;
const IDS = ['question','problem','symmetry','blindness','machinery','layer','lmax','forces','locality','hull','genealogy'];
const seen = [];
for (const f of ['ui','e3','linalg','app']) {
  try { await import(`${base}/tutorial/js/${f}.js`); }
  catch(e){ if (!domish(e.message)) { console.log('PARSE ERR', f, e.message.split('\n')[0]); bad++; } }
}
for (const f of fs.readdirSync(`${base}/tutorial/js/chapters`).sort()) {
  try {
    const m = await import(`${base}/tutorial/js/chapters/${f}`);
    if (!m.default?.id) { console.log('SHAPE ERR', f, 'no default.id'); bad++; }
    else { seen.push(m.default.id); if (typeof m.default.render !== 'function') { console.log('SHAPE ERR', f, 'no render()'); bad++; } }
  } catch(e){ if (!domish(e.message)) { console.log('PARSE ERR', f, e.message.split('\n')[0]); bad++; } }
}
const missing = IDS.filter(i => !seen.includes(i));
const extra = seen.filter(i => !IDS.includes(i));
if (missing.length) { console.log('MISSING IDS:', missing.join(', ')); bad++; }
if (extra.length) { console.log('UNEXPECTED IDS:', extra.join(', ')); bad++; }
console.log(bad === 0 ? `ALL MODULES OK (${seen.length} chapters: ${seen.join(', ')})` : `${bad} PROBLEM(S)`);
