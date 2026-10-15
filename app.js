// Decorative only: deterministic positions, no event interception or rendering loop.
const starfield = document.querySelector('#starfield');
if (starfield) {
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < 64; i++) {
    const star = document.createElement('i');
    star.className = 'star';
    star.style.cssText = [
      'left:' + ((i * 47 + 13) % 100) + '%',
      'top:' + ((i * 31 + 7) % 100) + '%',
      '--size:' + (i % 4 === 0 ? 4 : 2.5) + 'px',
      '--opacity:' + (0.45 + (i % 4) * 0.12),
      '--duration:' + (7 + (i % 7)) + 's',
      '--delay:-' + ((i * 11) % 40) + 's',
      '--dx:' + ((i % 2 ? 1 : -1) * (90 + i % 60)) + 'px',
      '--dy:' + (-110 - i % 70) + 'px'
    ].join(';');
    if (i % 6 === 0) {
      const line = document.createElement('span');
      line.className = 'star-thread';
      line.style.setProperty('--angle', (i * 17 % 150 - 75) + 'deg');
      star.append(line);
    }
    fragment.append(star);
  }
  starfield.append(fragment);
  const syncVisibility = () => document.body.classList.toggle('page-hidden', document.hidden);
  document.addEventListener('visibilitychange', syncVisibility);
  syncVisibility();
}
const copyCitation = document.querySelector('#copy-citation');
if (copyCitation) {
  copyCitation.addEventListener('click', async () => {
    const code = document.querySelector('#citation-code');
    const status = document.querySelector('#citation-status');
    try {
      await navigator.clipboard.writeText(code.textContent);
      copyCitation.textContent = 'Copied!';
      status.textContent = 'Citation copied to clipboard.';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyCitation.textContent = 'Copy';
      status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });
}
const benchmarks = [
  ['EmbSpatial','Spatial & Relational','perception',81.68,82.53,'RynnBrain1.1-9B'],
  ['VSI','Spatial & Relational','perception',63.30,71.59,'RynnBrain1.1-9B'],
  ['ERQA','Spatial & Relational','perception',47.31,46.09,'Cosmos3-Nano'],
  ['OpenEQA','Scene & State','perception',63.33,65.85,'Mimo-Embodied-7B'],
  ['UniEQA','Scene & State','perception',50.66,56.62,'RynnBrain1.1-9B'],
  ['RoboVQA','Scene & State','perception',71.79,66.09,'Mimo-Embodied-7B'],
  ['Cosmos','Temporal & Event','perception',68.96,63.18,'Cosmos3-Nano'],
  ['ET-Plan','Long-Horizon Action Org.','planning',61.15,69.55,'Cosmos3-Nano'],
  ['EB-ALFRED','Long-Horizon Action Org.','planning',26.67,17.33,'Mimo-Embodied-7B'],
  ['EB-Habitat','Long-Horizon Action Org.','planning',37.00,26.67,'Cosmos3-Nano'],
  ['VABench-P','Point & Affordance','execution',63.33,47.54,'Cosmos3-Nano'],
  ['Where2Place','Point & Affordance','execution',70.79,75.48,'RynnBrain-8B']
];
const body = document.querySelector('#benchmark-table tbody');
benchmarks.forEach(([name,dimension,group,score,baseline,model]) => {
  const row = document.createElement('tr'); row.dataset.group = group;
  const delta = score - baseline;
  row.innerHTML = `<th scope="row">${name}<span class="benchmark-label">${dimension}</span></th><td class="${delta > 0 ? 'best' : ''}">${score.toFixed(2)}</td><td>${model}</td><td>${baseline.toFixed(2)}</td><td class="${delta > 0 ? 'gain' : 'loss'}">${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(2)}</td>`;
  body.append(row);
});
document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-filter]').forEach(b => {b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button));});
  body.querySelectorAll('tr').forEach(row => {row.hidden = button.dataset.filter !== 'all' && row.dataset.group !== button.dataset.filter;});
}));
const composition = [['Spatial & Relational',31.34],['Scene & State',32.61],['Temporal & Event',2.57],['Point & Affordance',23.92],['Long-Horizon Action',9.56]];
document.querySelector('#stage-a-bars').innerHTML = composition.map(([label,share]) => `<div class="mixture-row"><span>${label}</span><div class="bar"><i style="width:${share / 32.61 * 100}%"></i></div><span>${share.toFixed(2)}%</span></div>`).join('');
const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('.zoomable').forEach(button => button.addEventListener('click', () => {
  dialog.querySelector('img').src = button.dataset.image;
  dialog.querySelector('img').alt = button.querySelector('img').alt;
  dialog.showModal();
}));
dialog.querySelector('.close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {if (event.target === dialog) {const r=dialog.getBoundingClientRect(); if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom) dialog.close();}});
const observer = new IntersectionObserver(entries => {entries.forEach(entry => {if(entry.isIntersecting) document.querySelectorAll('nav a').forEach(a => a.classList.toggle('active',a.getAttribute('href') === '#' + entry.target.id));});},{rootMargin:'-15% 0px -60% 0px'});
document.querySelectorAll('main > section[id]').forEach(section=>observer.observe(section));
