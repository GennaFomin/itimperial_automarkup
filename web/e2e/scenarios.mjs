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
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message))
page.on('console', (m) => m.type() === 'error' && errors.push('console: ' + m.text()))

async function createTask(title, scenarioLabel) {
  await page.click('button:has-text("Новая задача")')
  await page.waitForSelector('.modal')
  await page.setInputFiles('input[type=file]', VIDEO)
  await page.waitForFunction(() => document.querySelector('.drop--filled') !== null)
  await page.fill('.modal .input', title)
  await page.selectOption('.modal select.select', { label: scenarioLabel })
  await page.click('button:has-text("В очередь на разметку")')
  await page.waitForSelector('.modal', { state: 'detached' })
}

await page.goto(BASE)
await page.waitForSelector('.tasks__title')

console.log('▶ Создаём по задаче на каждый сценарий контракта')
await createTask('Пустой результат', 'Пустой результат — ни одного сегмента')
await createTask('Часть keyframe потеряна', 'done_with_errors — часть keyframe не посчитана')
await createTask('Джоба падает', 'Джоба падает')
await createTask('Долгая обработка', 'Долгая обработка — дольше 10 минут')
await createTask('Обычный прогноз', 'Обычный прогноз')
console.log('  карточек:', await page.locator('.card').count())

await page.waitForTimeout(13000)
await page.screenshot({ path: `${SHOTS}/s1-list.png` })
console.log('  статусы:', (await page.locator('.status').allTextContents()).join(' | '))

const open = async (title) => {
  await page.locator('.card', { hasText: title }).locator('button:has-text("Открыть разметку")').click()
  await page.waitForTimeout(2500)
}

console.log('\n▶ Пустой результат')
await open('Пустой результат')
const emptyTitle = await page.locator('.empty__title').first().textContent().catch(() => null)
console.log('  панель показывает:', emptyTitle?.trim())
console.log('  сегментов:', await page.locator('.seg').count())
if (await page.locator('.tl__track').count() === 0) errors.push('ТАЙМЛАЙН НЕ ОТРИСОВАН ПРИ ПУСТОМ ПРОГНОЗЕ')
await page.screenshot({ path: `${SHOTS}/s2-empty.png` })

// Пустой прогноз — валидный результат: должна быть возможность разметить руками.
await page.keyboard.press('c')
const tb = await page.locator('.tl__track').boundingBox()
await page.mouse.move(tb.x + tb.width * 0.3, tb.y + 40)
await page.mouse.down()
await page.mouse.move(tb.x + tb.width * 0.45, tb.y + 40, { steps: 10 })
await page.mouse.up()
await page.waitForSelector('.rpick', { timeout: 5000 }).catch(() => errors.push('НЕЛЬЗЯ РАЗМЕТИТЬ ПУСТОЙ РОЛИК'))
const rpickTitle = await page.locator('.rpick__title').textContent()
console.log('  ручная разметка:', rpickTitle?.trim())
await page.click('.rpick .btn--primary')
await page.waitForTimeout(300)
console.log('  создано сегментов:', await page.locator('.seg').count())
if (await page.locator('.seg').count() !== 1) errors.push('РУЧНОЙ СЕГМЕНТ НЕ СОЗДАН')
await page.screenshot({ path: `${SHOTS}/s3-empty-manual.png` })
await page.click('a:has-text("Задачи")')
await page.waitForSelector('.tasks__title')

console.log('\n▶ done_with_errors — часть keyframe не посчитана')
await open('Часть keyframe потеряна')
const banner = await page.locator('.banner--warn').first().textContent().catch(() => null)
console.log('  баннер:', banner?.trim().slice(0, 90))
if (!banner) errors.push('НЕТ БАННЕРА ПРО ОШИБКИ ПАЙПЛАЙНА')
const noKf = await page.locator('.kf__warn').count()
console.log('  карточек без keyframe:', noKf)
if (noKf === 0) errors.push('НЕ ПОМЕЧЕНЫ СЕГМЕНТЫ БЕЗ KEYFRAME')
await page.waitForTimeout(3000)
await page.screenshot({ path: `${SHOTS}/s4-with-errors.png` })
await page.click('a:has-text("Задачи")')
await page.waitForSelector('.tasks__title')

console.log('\n▶ Упавшая джоба')
const failedCard = page.locator('.card', { hasText: 'Джоба падает' })
console.log('  статус карточки:', (await failedCard.locator('.status').textContent())?.trim())
console.log('  кнопка заблокирована:', await failedCard.locator('button:has-text("Ждём авторазметку")').isDisabled())

console.log('\n▶ Долгая обработка')
await page.goto(BASE)
await page.waitForSelector('.tasks__title')
const slowCard = page.locator('.card', { hasText: 'Долгая обработка' })
console.log('  статус:', (await slowCard.locator('.status').textContent())?.trim())
const hasProgress = await slowCard.locator('.progress__bar').count() > 0
console.log('  полоса прогресса:', hasProgress ? 'есть' : 'НЕТ')
if (!hasProgress) errors.push('НЕТ ПРОГРЕССА У ДОЛГОЙ ЗАДАЧИ')
await page.screenshot({ path: `${SHOTS}/s5-list-final.png` })

console.log('\n▶ Работа без видео: разметка остаётся полностью рабочей')
// Фикстуры не содержат ролика, поэтому плеер обязан честно сказать об этом,
// а не показывать битую картинку. Всё остальное должно работать как обычно.
await open('Обычный прогноз')
const missing = await page.locator('.stage__missing').count()
console.log('  плеер объясняет отсутствие видео:', missing > 0 ? 'да' : 'НЕТ')
if (!missing) errors.push('ПЛЕЕР МОЛЧА ПОКАЗЫВАЕТ ПУСТОТУ')
const segCount = await page.locator('.seg').count()
console.log('  таймлайн жив:', segCount, 'сегментов')
if (!segCount) errors.push('ТАЙМЛАЙН ПУСТ БЕЗ ВИДЕО')
// Правка без видео тоже должна проходить целиком.
await page.locator('.seg').nth(1).click()
await page.click('.panel__tab:has-text("Сегмент")')
await page.waitForSelector('.insp')
// Считаем по счётчику в шапке, а не по инспектору: клавиша отмечает сегмент и
// сразу переводит к следующему непроверенному, поэтому в инспекторе окажется
// уже другой сегмент — как и задумано.
const checkedBefore = await page.locator('.ed__stat-val').nth(3).textContent()
await page.keyboard.press('y')
await page.waitForTimeout(300)
const checkedAfter = await page.locator('.ed__stat-val').nth(3).textContent()
console.log(`  отметка проверки без видео: ${checkedBefore} → ${checkedAfter}`)
if (checkedBefore === checkedAfter) errors.push('ОТМЕТКА ПРОВЕРКИ НЕ РАБОТАЕТ')
await page.screenshot({ path: `${SHOTS}/s6-no-video.png` })

await browser.close()
console.log('\n' + (errors.length ? '❌ ПРОБЛЕМЫ:\n' + errors.map(e => '  - ' + e).join('\n') : '✅ Все сценарии прошли'))
process.exit(errors.length ? 1 : 0)
