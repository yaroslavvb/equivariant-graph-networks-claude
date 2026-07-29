import { h } from '../ui.js';
export default {
  id: 'genealogy',
  title: 'The genealogy, and where it lands',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow' }, 'Chapter'),
      h('h1', {}, 'The genealogy, and where it lands'),
      h('p', { class: 'lede' }, 'This chapter is still being written.'));
  },
};
