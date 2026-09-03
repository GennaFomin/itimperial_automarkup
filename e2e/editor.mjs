import { chromium } from 'playwright'
// Путь к Chromium: по умолчанию берём тот, что нашёл Playwright, но в средах
// с преднастроенным браузером его задаёт PLAYWRIGHT_CHROMIUM_PATH.
const CHROME = process.env.PLAYWRIGHT_CHROMIUM_PATH
const launchOptions = CHROME ? { executablePath: CHROME } : {}
const BASE = process.env.E2E_BASE_URL ?? 'http://localhost:5173/'
const [VIDEO, SHOTS] = process.argv.slice(2)
const errors = []

const browser = await chromium.launch(launchOptions)
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } })
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message))

const shot = async (name) => {
  await page.screenshot({ path: `${SHOTS}/${name}.png` })
  console.log('  📸', name)
}
const step = (s) => console.log('▶', s)

await page.goto(BASE)
await page.waitForSelector('.tasks__title')
step('Список задач — пусто')
await shot('01-empty')

step('Создание задачи')
await page.click('button:has-text("Новая задача")')
await page.waitForSelector('.modal')
await page.setInputFiles('input[type=file]', VIDEO)
await page.waitForFunction(() => document.querySelector('.drop--filled') !== null, { timeout: 15000 })
await page.fill('.input', 'Сборка узла, камера 2')
await shot('02-create')

await page.click('button:has-text("В очередь на разметку")')
await page.waitForSelector('.card')
step('Задача в очереди')
await shot('03-queued')

step('Ждём авторазметку')
await page.waitForSelector('button:has-text("Открыть разметку"):not([disabled])', { timeout: 30000 })
await shot('04-ready')

await page.click('button:has-text("Открыть разметку")')
await page.waitForSelector('.tl__track', { timeout: 20000 })
await page.waitForTimeout(2500)
const segCount = await page.locator('.seg').count()
console.log('  сегментов на таймлайне:', segCount)
if (segCount === 0) errors.push('НЕТ СЕГМЕНТОВ НА ТАЙМЛАЙНЕ')
step('Экран разметки')
await shot('05-editor')

step('Ключевые кадры извлекаются')
await page.waitForFunction(() => document.querySelectorAll('.kf__thumb img').length >= 3, { timeout: 30000 })
  .catch(() => errors.push('КАДРЫ НЕ ИЗВЛЕКЛИСЬ'))
const thumbs = await page.locator('.kf__thumb img').count()
console.log('  превью кадров готово:', thumbs)
await shot('06-keyframes')

step('Клик по ключевому кадру открывает инспектор')
await page.locator('.kf').nth(2).click()
await page.waitForSelector('.insp')
await shot('07-inspector')

step('Смена действия клавишей 2')
const before = await page.locator('.insp .opt--on').first().textContent()
await page.keyboard.press('2')
await page.waitForTimeout(200)
const after = await page.locator('.insp .opt--on').first().textContent()
console.log(`  действие: ${before?.trim()} → ${after?.trim()}`)

step('Зум колесом на таймлайне')
const spanBefore = await page.locator('.tl__zoom-label').textContent()
await page.locator('.tl__track').hover()
for (let i = 0; i < 6; i++) await page.mouse.wheel(0, -120)
await page.waitForTimeout(300)
const spanAfter = await page.locator('.tl__zoom-label').textContent()
console.log(`  окно: ${spanBefore?.trim()} → ${spanAfter?.trim()}`)
if (spanBefore === spanAfter) errors.push('ЗУМ КОЛЕСОМ НЕ РАБОТАЕТ')
await shot('08-zoomed')

step('Инструмент «Вырезать»: выделяем середину сегмента')
await page.keyboard.press('0')   // полный обзор: на узком окне диапазон был бы короче минимума
await page.waitForTimeout(200)
await page.keyboard.press('c')
// Вырезаем середину ОДНОГО сегмента: ожидаем ровно +2 (хвост, вырез, хвост).
const wide = await page.evaluate(() => {
  const best = [...document.querySelectorAll('.seg')]
    .map((el) => ({ el, w: el.getBoundingClientRect().width }))
    .sort((a, b) => b.w - a.w)[0]
  const r = best.el.getBoundingClientRect()
  return { x: r.x, y: r.y, w: r.width, h: r.height }
})
console.log(`  самый широкий сегмент: ${Math.round(wide.w)} px`)
await page.mouse.move(wide.x + wide.w * 0.32, wide.y + wide.h / 2)
await page.mouse.down()
await page.mouse.move(wide.x + wide.w * 0.68, wide.y + wide.h / 2, { steps: 12 })
await page.mouse.up()
await page.waitForSelector('.rpick', { timeout: 5000 }).catch(() => errors.push('ПИКЕР ВЫРЕЗА НЕ ОТКРЫЛСЯ'))
await shot('09-carve-picker')

const countBefore = await page.locator('.seg').count()
await page.click('.rpick button:has-text("Вырезать")')
await page.waitForTimeout(400)
const countAfter = await page.locator('.seg').count()
console.log(`  сегментов: ${countBefore} → ${countAfter} (ожидалось +2)`)
if (countAfter !== countBefore + 2) errors.push(`ВЫРЕЗ ДАЛ ${countAfter}, ОЖИДАЛОСЬ ${countBefore + 2}`)
await shot('10-carved')

step('Разрез клавишей S')
await page.keyboard.press('v')
await page.locator('.seg').nth(3).click()
await page.waitForTimeout(200)
const cntPreSplit = await page.locator('.seg').count()
await page.keyboard.press('s')
await page.waitForTimeout(300)
console.log(`  после split: ${cntPreSplit} → ${await page.locator('.seg').count()}`)

step('Undo / Redo')
const cnt1 = await page.locator('.seg').count()
await page.keyboard.press('Control+z')
await page.waitForTimeout(250)
const cnt2 = await page.locator('.seg').count()
await page.keyboard.press('Control+Shift+z')
await page.waitForTimeout(250)
const cnt3 = await page.locator('.seg').count()
console.log(`  ${cnt1} →undo ${cnt2} →redo ${cnt3}`)
if (cnt1 === cnt2) errors.push('UNDO НЕ СРАБОТАЛ')
if (cnt3 !== cnt1) errors.push('REDO НЕ ВЕРНУЛ СОСТОЯНИЕ')

step('Воспроизведение и скорость 2×')
await page.click('.transport__play')
await page.waitForTimeout(1200)
await page.click('.transport__speed:has-text("2×")')
await page.waitForTimeout(800)
const tc = await page.locator('.stage__tc').textContent()
console.log('  таймкод во время игры:', tc?.trim())
if (tc?.trim() === '00:00.000') errors.push('ВИДЕО НЕ ИГРАЕТ')
await page.click('.transport__play')
await shot('11-playing')

step('Экспорт')
await page.click('button:has-text("Экспорт")')
await page.waitForTimeout(200)
const [dl] = await Promise.all([
  page.waitForEvent('download', { timeout: 10000 }),
  page.click('button:has-text("CSV — правка")'),
])
const path = await dl.path()
const csv = (await import('node:fs')).readFileSync(path, 'utf8')
// BOM проверяем на сыром содержимом: trim() съел бы \uFEFF как пробельный символ.
const hasBom = csv.charCodeAt(0) === 0xfeff
if (!hasBom) errors.push('В CSV НЕТ BOM')
const lines = csv.slice(hasBom ? 1 : 0).trim().split('\n')
console.log('  CSV строк:', lines.length, '| BOM:', hasBom ? 'есть' : 'НЕТ')
console.log('  заголовок:', lines[0])
console.log('  пример:   ', lines[1])

step('Отправка review')
await page.click('button:has-text("Отправить review")')
await page.waitForSelector('button:has-text("Сохранено")', { timeout: 10000 })
  .catch(() => errors.push('REVIEW НЕ СОХРАНИЛСЯ'))
await shot('12-saved')

await browser.close()
console.log('\n' + (errors.length ? '❌ ПРОБЛЕМЫ:\n' + errors.map(e => '  - ' + e).join('\n') : '✅ Все шаги прошли'))
process.exit(errors.length ? 1 : 0)
