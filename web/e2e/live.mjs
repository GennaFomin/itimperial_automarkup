import { chromium } from 'playwright'
const CHROME = process.env.PLAYWRIGHT_CHROMIUM_PATH
const opts = CHROME ? { executablePath: CHROME } : {}
const [VIDEO, SHOTS] = process.argv.slice(2)

// Ролик для этого прогона должен быть в VP9, а не в привычном H.264: Chromium,
// который поставляет Playwright, собран без проприетарных кодеков и H.264 не
// проигрывает (canPlayType возвращает пустую строку). На настоящих браузерах
// ограничения нет — оно только у тестовой сборки.
const errors = []

const browser = await chromium.launch(opts)
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } })
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message))
page.on('console', (m) => m.type() === 'error' && errors.push('console: ' + m.text()))
const shot = async (n) => { await page.screenshot({ path: `${SHOTS}/${n}.png` }); console.log('  📸', n) }

await page.goto('http://127.0.0.1:8000/')
await page.waitForSelector('.tasks__title')
console.log('▶ список задач с сервера:', await page.locator('.card').count(), 'карточек')
await shot('L1-list')

console.log('▶ создаём задачу настоящей загрузкой')
await page.click('button:has-text("Новая задача")')
await page.setInputFiles('input[type=file]', VIDEO)
await page.waitForFunction(() => document.querySelector('.drop--filled') !== null, { timeout: 15000 })
await page.fill('.modal .input', 'Живой прогон')
await shot('L2-create')
await page.click('button:has-text("В очередь на разметку")')
await page.waitForSelector('.modal', { state: 'detached' })

// Открываем именно свою задачу, а не первую готовую в списке: соседние ролики
// могут быть от прошлых прогонов и с другим содержимым.
const card = page.locator('.card', { hasText: 'Живой прогон' })
await card.locator('button:has-text("Открыть разметку"):not([disabled])').waitFor({ timeout: 60000 })
console.log('▶ авторазметка готова')
await card.locator('button:has-text("Открыть разметку")').click()
await page.waitForSelector('.tl__track', { timeout: 30000 })
await page.waitForTimeout(2500)
const segs = await page.locator('.seg').count()
console.log('  сегментов из пайплайна:', segs)
if (!segs) errors.push('НЕТ СЕГМЕНТОВ')

const kf = await page.locator('.kf__thumb img').count()
console.log('  превью кадров с сервера:', kf)
if (!kf) errors.push('КАДРЫ С СЕРВЕРА НЕ ПРИШЛИ')

// Ждём именно готовности метаданных: сразу после вставки <video> readyState
// всегда 0, и мгновенная проверка ловила бы не загрузку, а гонку.
const videoOk = await page
  .waitForFunction(() => {
    const v = document.querySelector('video')
    return v && v.readyState >= 1 ? { src: v.getAttribute('src'), ready: v.readyState, dur: v.duration } : null
  }, { timeout: 20000 })
  .then((h) => h.jsonValue())
  .catch(() => null)
if (videoOk) console.log('  видео:', videoOk.src, '| readyState:', videoOk.ready, '| длина:', videoOk.dur?.toFixed(1), 'с')
else errors.push('ВИДЕО С СЕРВЕРА НЕ ЗАГРУЗИЛОСЬ')
await shot('L3-editor')

console.log('▶ отметка проверки')
await page.locator('.kf').first().click()
await page.waitForSelector('.insp')
const before = await page.locator('.ed__stat-val').nth(3).textContent()
await page.keyboard.press('y')
await page.waitForTimeout(300)
const after = await page.locator('.ed__stat-val').nth(3).textContent()
console.log(`  проверено: ${before} → ${after}`)
if (before === after) errors.push('ОТМЕТКА ПРОВЕРКИ НЕ СРАБОТАЛА')

console.log('▶ правка действия и вырез')
await page.keyboard.press('2'); await page.waitForTimeout(200)
await page.keyboard.press('0'); await page.waitForTimeout(200)
await page.keyboard.press('c')
const wide = await page.evaluate(() => {
  const b = [...document.querySelectorAll('.seg')].map(el => ({el, w: el.getBoundingClientRect().width})).sort((a,b)=>b.w-a.w)[0]
  const r = b.el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}
})
await page.mouse.move(wide.x + wide.w*0.35, wide.y + wide.h/2)
await page.mouse.down()
await page.mouse.move(wide.x + wide.w*0.65, wide.y + wide.h/2, { steps: 10 })
await page.mouse.up()
await page.waitForSelector('.rpick', { timeout: 5000 }).catch(() => errors.push('ПИКЕР НЕ ОТКРЫЛСЯ'))
const cBefore = await page.locator('.seg').count()
await page.click('.rpick .btn--primary')
await page.waitForTimeout(400)
console.log(`  сегментов: ${cBefore} → ${await page.locator('.seg').count()}`)
await shot('L4-edited')

console.log('▶ отправка правки на сервер')
await page.click('button:has-text("Отправить правку")')
await page.waitForSelector('button:has-text("Сохранено")', { timeout: 15000 })
  .catch(() => errors.push('ПРАВКА НЕ СОХРАНИЛАСЬ'))
await shot('L5-saved')

console.log('▶ повторное открытие: правка на месте, а не прогноз')
// Самый дорогой из возможных дефектов: если редактор откроется на прогнозе,
// следующее сохранение сотрёт работу человека — на задаче, которую список уже
// отмечает проверенной.
const editedActions = await page.locator('.seg').evaluateAll((els) =>
  els.map((el) => el.getAttribute('title')),
)
await page.reload()
await page.waitForSelector('.tl__track', { timeout: 30000 })
await page.waitForTimeout(2000)
const reopened = await page.locator('.seg').evaluateAll((els) =>
  els.map((el) => el.getAttribute('title')),
)
console.log(`  сегментов до перезагрузки: ${editedActions.length}, после: ${reopened.length}`)
if (reopened.length !== editedActions.length) errors.push('ПОСЛЕ ПЕРЕОТКРЫТИЯ ПРАВКА ПОТЕРЯНА')
const checkedAfter = await page.locator('.ed__stat-val').nth(3).textContent()
console.log('  отметка проверки пережила перезагрузку:', checkedAfter)
if (checkedAfter?.startsWith('0/')) errors.push('ОТМЕТКА ПРОВЕРКИ НЕ СОХРАНИЛАСЬ')
await shot('L6-reopened')

await browser.close()
console.log('\n' + (errors.length ? '❌ ПРОБЛЕМЫ:\n' + errors.map(e=>'  - '+e).join('\n') : '✅ Связка UI ↔ пайплайн работает'))
process.exit(errors.length ? 1 : 0)
