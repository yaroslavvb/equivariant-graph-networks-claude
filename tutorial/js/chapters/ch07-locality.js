import { h } from '../ui.js';
export default {
  id: 'locality',
  title: 'How far does locality reach',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow' }, 'Chapter'),
      h('h1', {}, 'How far does locality reach'),
      h('p', { class: 'lede' }, 'This chapter is still being written.'));
  },
};
