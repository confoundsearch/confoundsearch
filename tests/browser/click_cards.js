const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto('file://' + process.argv[2]);
  const out = {};
  const rgb = hex => 'rgb(' + [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)).join(', ') + ')';
  const count = sel => page.$$eval(sel, els => els.length);
  const text = sel => page.$eval(sel, e => e.innerText);
  out.cards = await count('.md-card');
  out.selectedAtStart = await count('.dag-svg.has-sel');
  const shown = () => page.$$eval('.dag-count', els => els.filter(e => getComputedStyle(e).display !== 'none').length);
  out.countsAtStart = await shown();

  // an arrow: only that edge is strong, its count becomes a pill, and the evidence names the edge
  const card = (await page.$$('.md-card'))[0];
  const name = await card.getAttribute('data-md');
  const color = await card.getAttribute('data-color');
  out.name = name;
  await page.click(`.leg-btn[data-md="${name.replace(/"/g, '\\"')}"][data-leg="to_dv"]`);
  out.edgeSelected = await count('.dag-svg line.sel');
  out.pillShown = await page.$$eval('.dag-count.pill.sel', els => els.map(e => getComputedStyle(e).display));
  out.countsOnEdge = await shown();
  out.fadedEdges = await page.$$eval('.dag-svg line[data-md]:not(.sel)', els => els.every(e => parseFloat(getComputedStyle(e).opacity) < 0.2));
  out.header = await text('#evidence-display .edge-head');
  out.pillOutline = await page.$eval('#evidence-display .var-pill', e => getComputedStyle(e).borderTopColor);
  out.panelBar = await page.$eval('#evidence-display', e => getComputedStyle(e).borderLeftColor);
  out.ringColor = await page.$eval('.dag-var.sel', e => e.getAttribute('stroke'));
  out.expectedRgb = rgb(color);
  out.color = color;
  out.legActive = await count('.leg-btn.active');

  // "also": the confound's other edge
  await page.click('#evidence-display .also a');
  out.otherHeader = await text('#evidence-display .edge-head');

  // a card: both edges, one after the other
  await page.click(`.md-card[data-md="${name.replace(/"/g, '\\"')}"] .card-name`);
  out.sections = await count('#evidence-display .edge-section');
  out.edgesBoth = await count('.dag-svg line.sel');
  out.countsOnBoth = await shown();
  out.ringsBoth = await count('.dag-var.sel');

  // a card past the five drawn: evidence, nothing in the diagram, nothing hidden
  const late = (await page.$$('.md-card')).slice(-1)[0];
  out.lateName = await late.getAttribute('data-md');
  await late.click();
  out.lateEvidence = (await text('#evidence-display .chip')).slice(0, 40);
  out.lateEdges = await count('.dag-svg line.sel');
  out.nodesStillThere = await count('.dag-md-circle');

  // the background: back to nothing chosen
  await page.click('.dag-svg', { position: { x: 5, y: 5 } });
  out.clearedSelection = await count('.dag-svg.has-sel');
  out.countsCleared = await shown();
  out.clearedPanel = (await text('#evidence-display')).slice(0, 30);
  out.errors = errors;
  console.log(JSON.stringify(out));
  await browser.close();
})();
