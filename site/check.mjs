/**
 * What the landing page has to do, checked in a browser rather than by eye.
 *
 *   python site/build.py
 *   python -m http.server 8791 --directory site/dist &
 *   npm install --no-save playwright        # not a project dependency: see below
 *   node site/check.mjs
 *
 * Playwright is deliberately not in package.json. `npm ci` runs on the publishing
 * path, and putting a browser download on it to serve a check that nothing in CI runs
 * would be paying for this on every deploy.
 *
 * The claims worth making automatically are the ones an eye is bad at. Whether the
 * first click makes a sound is one: a page that boots its audio graph, reports
 * "running" and stays silent looks exactly like one that works. So does a panel that
 * has scrolled its own keys out of a frame nobody thought to resize.
 */

import { chromium } from 'playwright';

const base = process.argv[2] || 'http://127.0.0.1:8791';
// CHROMIUM overrides the browser, for an environment that ships one already rather
// than letting playwright fetch its own.
const browser = await chromium.launch({
  ...(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {}),
  args: ['--autoplay-policy=no-user-gesture-required'],
});
let bad = 0;
const fail = (m) => { console.log('FAIL ' + m); bad++; };
const ok = (m) => console.log('ok   ' + m);

// ---- phone viewport, landing page ----
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
const page = await ctx.newPage();
const errs = [];
page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
page.on('pageerror', e => errs.push('pageerror: ' + e.message));
await page.goto(base + '/index.html', { waitUntil: 'load' });

const mode = await page.evaluate(() => document.compatMode);
mode === 'CSS1Compat' ? ok('landing compatMode ' + mode) : fail('landing compatMode ' + mode);

const overflow = await page.evaluate(() =>
  document.documentElement.scrollWidth - document.documentElement.clientWidth);
overflow <= 1 ? ok('no horizontal overflow at 390px') : fail('horizontal overflow ' + overflow + 'px');

const h1 = await page.textContent('h1');
ok('h1: ' + h1.trim());

const cards = await page.$$eval('.card h3', els => els.map(e => e.textContent));
cards.length === 6 ? ok('6 cards: ' + cards.join(', ')) : fail('cards: ' + cards.length);

// ---- the first click, inside the frame ----
const frame = page.frameLocator('#stage');
const btn = frame.locator('#start');
await btn.waitFor({ timeout: 15000 });
const label = await btn.textContent();
label.trim() === 'Play' ? ok('button reads "Play"') : fail('button reads "' + label + '"');

// the hero button, which is the one a phone visitor actually sees
const hero = page.locator('#hero-play');
await page.waitForFunction(() => {
  const b = document.getElementById('hero-play');
  return b && !b.disabled;
}, null, { timeout: 15000 });
const heroText = (await hero.textContent()).trim();
heroText === 'Play a synth now' ? ok('hero button keeps its own label')
                                : fail('hero label is "' + heroText + '"');
const heroBox = await hero.boundingBox();
heroBox.y + heroBox.height < 844
  ? ok(`hero Play is above the fold at 390x844 (y=${Math.round(heroBox.y)})`)
  : fail(`hero Play is below the fold (y=${Math.round(heroBox.y)})`);
await hero.click();
const f = page.frames().find(f => f.url().includes('chorus-polysynth'));
await f.waitForFunction(() => window.synth && window.synth.node, null, { timeout: 20000 });
ok('audio graph up');

const rms = await f.evaluate(async () => {
  const { ctx, node } = window.synth;
  const an = ctx.createAnalyser();
  an.fftSize = 2048;
  node.connect(an);
  const buf = new Float32Array(an.fftSize);
  let peak = 0;
  for (let i = 0; i < 40; i++) {
    await new Promise(r => setTimeout(r, 50));
    an.getFloatTimeDomainData(buf);
    let s = 0;
    for (const v of buf) s += v * v;
    peak = Math.max(peak, Math.sqrt(s / buf.length));
  }
  return peak;
});
rms > 0.01 ? ok('demo plays unprompted, peak RMS ' + rms.toFixed(4))
           : fail('silence after the first click, peak RMS ' + rms.toFixed(6));

const lit = await f.evaluate(() => document.querySelectorAll('.key.on').length);
ok('keys lit during demo: ' + lit);

const heroLabel = await hero.textContent();
heroLabel.trim() === 'Play again' ? ok('hero button mirrors "Play again"')
                                  : fail('hero button reads "' + heroLabel + '"');
const label2 = await btn.textContent();
label2.trim() === 'Play again' ? ok('button becomes "Play again"') : fail('button: ' + label2);
const disabled = await btn.isDisabled();
disabled ? fail('replay button is disabled') : ok('replay button stays live');

// ---- the demo yields to a real key ----
await f.evaluate(() => { window.synth.playDemo(); });
await page.waitForTimeout(300);
await f.evaluate(() => { window.synth.noteOn(72, 100); });
await page.waitForTimeout(120);
const stillDemo = await f.evaluate(() => document.querySelectorAll('.key.on').length);
ok('after playDemo + manual note, keys lit: ' + stillDemo);

// ---- the frame is sized to its content, not scrolling internally ----
const fit = await page.evaluate(() => {
  const el = document.getElementById('stage');
  const d = el.contentDocument;
  return { frame: Math.round(el.getBoundingClientRect().height),
           doc: d.documentElement.scrollHeight };
});
Math.abs(fit.frame - fit.doc) < 24
  ? ok(`frame fits content (${fit.frame} vs ${fit.doc})`)
  : fail(`frame ${fit.frame}px against content ${fit.doc}px: it will scroll internally`);

// ---- an instrument page on its own ----
const p2 = await ctx.newPage();
p2.on('pageerror', e => errs.push('patch pageerror: ' + e.message));
await p2.goto(base + '/p/plucky-fm-bass.html', { waitUntil: 'load' });
const m2 = await p2.evaluate(() => document.compatMode);
m2 === 'CSS1Compat' ? ok('patch page compatMode ' + m2) : fail('patch compatMode ' + m2);
const desc = await p2.getAttribute('meta[name=description]', 'content');
desc ? ok('patch meta description injected') : fail('no meta description on patch page');
await p2.click('#start');
await p2.waitForFunction(() => window.synth && window.synth.node, null, { timeout: 20000 });
const rms2 = await p2.evaluate(async () => {
  const { ctx, node } = window.synth;
  const an = ctx.createAnalyser(); an.fftSize = 2048; node.connect(an);
  const buf = new Float32Array(an.fftSize); let peak = 0;
  for (let i = 0; i < 40; i++) {
    await new Promise(r => setTimeout(r, 50));
    an.getFloatTimeDomainData(buf);
    let s = 0; for (const v of buf) s += v * v;
    peak = Math.max(peak, Math.sqrt(s / buf.length));
  }
  return peak;
});
rms2 > 0.01 ? ok('fm-bass riff plays, peak RMS ' + rms2.toFixed(4))
            : fail('fm-bass silent, peak RMS ' + rms2.toFixed(6));

errs.length ? fail('console errors:\n  ' + errs.join('\n  ')) : ok('no console errors');
await browser.close();
console.log(bad ? `\n${bad} FAILED` : '\nall checks passed');
process.exit(bad ? 1 : 0);
