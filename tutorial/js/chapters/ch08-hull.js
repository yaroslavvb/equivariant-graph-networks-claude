import { h } from '../ui.js';
export default {
  id: 'hull',
  title: 'Stability is a decision, not a regression',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow' }, 'Chapter'),
      h('h1', {}, 'Stability is a decision, not a regression'),
      h('p', { class: 'lede' }, 'This chapter is still being written.'));
  },
};
