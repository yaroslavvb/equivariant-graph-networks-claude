import { h } from '../ui.js';
export default {
  id: 'lmax',
  title: 'The l-max ablation',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow' }, 'Chapter'),
      h('h1', {}, 'The l-max ablation'),
      h('p', { class: 'lede' }, 'This chapter is still being written.'));
  },
};
