// Записываем анимированную страницу как webm — так получаем настоящий видеофайл
// без полноценного ffmpeg в окружении.
import { chromium } from 'playwright'
import { mkdirSync, readdirSync, renameSync } from 'node:fs'
import { join } from 'node:path'

const outDir = process.argv[2]
mkdirSync(outDir, { recursive: true })

const browser = await chromium.launch(launchOptions)
const ctx = await browser.newContext({
  viewport: { width: 640, height: 360 },
  recordVideo: { dir: outDir, size: { width: 640, height: 360 } },
})
const page = await ctx.newPage()
await page.setContent(`
<style>
  body { margin:0; height:100vh; overflow:hidden; font-family:sans-serif; transition:background .2s; }
  #dot { position:absolute; width:48px; height:48px; border-radius:8px; background:#ff5a1f; }
  #tc { position:absolute; left:12px; top:10px; color:#fff; font-size:34px; font-weight:800; font-variant-numeric:tabular-nums; }
  #ph { position:absolute; right:12px; top:10px; color:#fff; font-size:20px; opacity:.85; }
</style>
<div id="dot"></div><div id="tc">0.0</div><div id="ph"></div>
<script>
  const COLORS = ['#16202c','#3b1f18','#123330','#241d3a','#332f14','#1b2f1b'];
  const NAMES = ['подойти','взять','переместить','совместить','прижать','отпустить'];
  const t0 = performance.now();
  function frame() {
    const t = (performance.now() - t0) / 1000;
    const phase = Math.floor(t / 5) % 6;
    document.body.style.background = COLORS[phase];
    document.getElementById('tc').textContent = t.toFixed(1);
    document.getElementById('ph').textContent = NAMES[phase];
    const p = (t % 5) / 5;
    document.getElementById('dot').style.left = (p * 560 + 20) + 'px';
    document.getElementById('dot').style.top = (150 + Math.sin(t * 2.2) * 90) + 'px';
    requestAnimationFrame(frame);
  }
  frame();
</script>`)

await page.waitForTimeout(40_000)
await ctx.close()
await browser.close()

const file = readdirSync(outDir).find((f) => f.endsWith('.webm'))
renameSync(join(outDir, file), join(outDir, 'clip.webm'))
console.log('готово:', join(outDir, 'clip.webm'))
