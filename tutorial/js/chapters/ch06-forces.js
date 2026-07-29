import { h } from '../ui.js';
export default {
  id: 'forces',
  title: 'Forces from a gradient',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow' }, 'Chapter'),
      h('h1', {}, 'Forces from a gradient'),
      h('p', { class: 'lede' }, 'This chapter is still being written.'));
  },
};
