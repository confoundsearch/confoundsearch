// The walkthrough in a real browser: it opens on its own, steps forwards and back, focuses one part of the
// page at a time, points at the example arrow last, and ends on Escape or when that arrow is clicked.
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1300, height: 820 } });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto('file://' + process.argv[2]);
  const out = {};
  const focus = () => page.$$eval('[data-tour].tour-focus', els => els.map(e => e.getAttribute('data-tour')));
  const dot = () => page.$$eval('.tour-dot', els => els.findIndex(e => e.classList.contains('on')));
  const blurred = () => page.$$eval('[data-tour]:not(.tour-focus)', els => els.every(e => getComputedStyle(e).filter.includes('blur')));
  out.visibleAtStart = await page.isVisible('#tour-card');
  out.dots = await page.$$eval('.tour-dot', els => els.length);
  // a touch mouse or trackpad scrolls by wheel events: with the tour open the page (and graph) stays put
  await page.setViewportSize({ width: 1300, height: 360 });
  await page.mouse.move(650, 200);
  await page.mouse.wheel(0, 600);
  await page.waitForTimeout(150);
  out.scrolledDuringTour = await page.evaluate(() => window.scrollY);
  await page.setViewportSize({ width: 1300, height: 820 });
  out.step1 = { dot: await dot(), focus: await focus(), blurred: await blurred(), back: await page.isDisabled('.tour-prev') };
  await page.click('.tour-next');
  out.step2 = { dot: await dot(), focus: await focus(), blurred: await blurred() };
  await page.keyboard.press('ArrowRight');
  out.step3 = { dot: await dot(), focus: await focus() };
  await page.keyboard.press('ArrowLeft');
  out.backToStep2 = await dot();
  await page.click('.tour-dot:nth-child(4)');
  out.step4links = await page.$$eval('.tour-text a', els => els.map(e => [e.innerText, e.target]));
  await page.click('.tour-dot:nth-child(5)');
  out.step5 = {
    dot: await dot(),
    focus: await focus(),
    next: (await page.innerText('.tour-next')).trim(),
    chosen: await page.$$eval('.dag-svg line.sel', els => els.map(e => e.getAttribute('data-md') + '|' + e.getAttribute('data-leg'))),
    pulsing: await page.$$eval('.dag-svg line', els => els.filter(e => getComputedStyle(e).animationName !== 'none').length),
    links: await page.$$eval('.tour-text a', els => els.map(e => [e.innerText, e.target])),
    text: await page.innerText('.tour-text'),
  };
  await page.keyboard.press('Escape');
  await page.setViewportSize({ width: 1300, height: 360 });
  await page.mouse.move(650, 200);
  await page.mouse.wheel(0, 600);
  await page.waitForTimeout(150);
  out.scrolledAfterTour = await page.evaluate(() => window.scrollY);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.setViewportSize({ width: 1300, height: 820 });
  out.closedByEscape = !(await page.isVisible('#tour-card')) && (await page.$$eval('body.tour-on', e => e.length)) === 0;
  await page.click('#tour-open');
  out.reopenedAtStep1 = (await page.isVisible('#tour-card')) && (await dot()) === 0;
  await page.click('.tour-dot:nth-child(5)');
  const chosen = await page.$('.dag-svg line.sel');
  const box = await chosen.boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(200);
  out.endedByClickingTheArrow = !(await page.isVisible('#tour-card'));
  out.evidenceAfterClick = (await page.innerText('#evidence-display .edge-head')).replace(/\s+/g, ' ');
  // hovering an arrow darkens it; the wide invisible line takes the pointer a little off the visible one
  await page.click('.dag-svg', { position: { x: 5, y: 5 } });
  const line = await page.$('.dag-svg line[data-md][data-leg="to_iv"]');
  const before = await line.evaluate(e => getComputedStyle(e).stroke);
  const b = await line.boundingBox();
  const ends = await line.evaluate(e => ['x1', 'y1', 'x2', 'y2'].map(k => parseFloat(e.getAttribute(k))));
  const len = Math.hypot(ends[2] - ends[0], ends[3] - ends[1]);
  const off = 6;  // across the line, so the pointer is beside the visible stroke but inside the wide one
  await page.mouse.move(b.x + b.width / 2 - off * (ends[3] - ends[1]) / len, b.y + b.height / 2 + off * (ends[2] - ends[0]) / len);
  out.hover = { before, after: await line.evaluate(e => getComputedStyle(e).stroke) };
  out.errors = errors;
  console.log(JSON.stringify(out));
  await browser.close();
})();
