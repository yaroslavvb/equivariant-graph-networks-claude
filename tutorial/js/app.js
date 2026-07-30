// Tutorial shell: chapter registry, routing, keyboard navigation, shared helpers.

import ch00 from './chapters/ch00-question.js';
import ch01 from './chapters/ch01-problem.js';
import ch02 from './chapters/ch02-symmetry.js';
import ch03 from './chapters/ch03-blindness.js';
import ch04 from './chapters/ch04-machinery.js';
import ch05 from './chapters/ch05-couplings.js';
import ch06 from './chapters/ch06-layer.js';
import ch07 from './chapters/ch07-lmax.js';
import ch08 from './chapters/ch08-forces.js';
import ch09 from './chapters/ch09-locality.js';
import ch10 from './chapters/ch10-hull.js';
import ch11 from './chapters/ch11-genealogy.js';

const CHAPTERS = [ch00, ch01, ch02, ch03, ch04, ch05, ch06, ch07, ch08, ch09, ch10, ch11];

// Published for chapters that want to cross-reference by id rather than by a
// number that changes whenever the running order does.
window.CHAPTER_NUMBER = Object.fromEntries(CHAPTERS.map((c, i) => [c.id, i]));

const tocList = document.getElementById('tocList');
const stage = document.getElementById('stage');
const toc = document.getElementById('toc');

const rendered = new Set();
let current = 0;

CHAPTERS.forEach((ch, i) => {
  const li = document.createElement('li');
  const a = document.createElement('a');
  a.href = `#${ch.id}`;
  a.textContent = ch.title;
  a.dataset.index = String(i);
  li.appendChild(a);
  tocList.appendChild(li);

  const sec = document.createElement('section');
  sec.className = 'chapter';
  sec.id = `chapter-${ch.id}`;
  stage.appendChild(sec);
});

function chapterNav(i) {
  const nav = document.createElement('div');
  nav.className = 'chapnav';
  const prev = document.createElement('button');
  prev.textContent = '← Previous';
  prev.disabled = i === 0;
  prev.onclick = () => go(i - 1);
  const next = document.createElement('button');
  next.className = 'primary';
  next.textContent = i === CHAPTERS.length - 1 ? 'Back to the top' : 'Next →';
  next.onclick = () => go(i === CHAPTERS.length - 1 ? 0 : i + 1);
  const prog = document.createElement('span');
  prog.className = 'progress';
  prog.innerHTML = `${i + 1} / ${CHAPTERS.length} &nbsp;·&nbsp; <span class="kbd">←</span> <span class="kbd">→</span> to move`;
  const spacer = document.createElement('span');
  spacer.className = 'spacer';
  nav.append(prev, prog, spacer, next);
  return nav;
}

async function go(i, { push = true } = {}) {
  if (i < 0 || i >= CHAPTERS.length) return;
  current = i;
  const ch = CHAPTERS[i];

  document.querySelectorAll('.chapter').forEach((s) => s.classList.remove('active'));
  const sec = document.getElementById(`chapter-${ch.id}`);
  sec.classList.add('active');

  tocList.querySelectorAll('a').forEach((a, k) => {
    a.classList.toggle('active', k === i);
    if (rendered.has(CHAPTERS[k].id)) a.classList.add('done');
  });

  if (!rendered.has(ch.id)) {
    const body = document.createElement('div');
    sec.appendChild(body);
    try {
      await ch.render(body);
      // The shell owns the chapter number, so inserting or reordering a chapter
      // cannot leave a stale literal behind in the module.
      const eyebrow = body.querySelector('.eyebrow');
      if (eyebrow) eyebrow.textContent = i === 0 ? 'Overture' : `Chapter ${i}`;
    } catch (err) {
      body.innerHTML = `<div class="note dat"><span class="tag">this demo failed to load</span>
        <code>${String(err && err.message ? err.message : err)}</code></div>`;
      console.error(`chapter ${ch.id}:`, err);
    }
    sec.appendChild(chapterNav(i));
    rendered.add(ch.id);
    tocList.querySelectorAll('a')[i].classList.add('done');
  }

  if (push && location.hash !== `#${ch.id}`) history.replaceState(null, '', `#${ch.id}`);
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
  toc.classList.remove('open');
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([sec]).catch(() => {});
  }
}

tocList.addEventListener('click', (e) => {
  const a = e.target.closest('a');
  if (!a) return;
  e.preventDefault();
  go(Number(a.dataset.index));
});

document.getElementById('menuBtn').onclick = () => toc.classList.toggle('open');

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea, select')) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'ArrowRight') { e.preventDefault(); go(current + 1); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); go(current - 1); }
});

window.addEventListener('hashchange', () => {
  const i = CHAPTERS.findIndex((c) => c.id === location.hash.slice(1));
  if (i >= 0 && i !== current) go(i, { push: false });
});

const start = CHAPTERS.findIndex((c) => c.id === location.hash.slice(1));
go(start >= 0 ? start : 0, { push: false });
