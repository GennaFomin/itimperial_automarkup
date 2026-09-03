/**
 * Тестовый ролик для сквозных прогонов.
 *
 * Должен проходить требования кейса, иначе диалог создания задачи отклонит его
 * ещё до отправки: контейнер mp4 или mov, не длиннее тридцати секунд, высота от
 * 720. Прежний вариант — запись анимированной страницы в webm 640×360 на сорок
 * секунд — нарушал все три и годился только когда UI работал без бэкенда.
 *
 * Кодек внутри mp4 — VP9, а не привычный H.264: Chromium, который поставляет
 * Playwright, собран без проприетарных кодеков и H.264 не проигрывает
 * (`canPlayType` возвращает пустую строку). На настоящих браузерах такого
 * ограничения нет — это свойство тестовой сборки, а не приложения.
 */
import { execFileSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { join } from 'node:path'

const outDir = process.argv[2]
const seconds = Number(process.argv[3] ?? 12)
mkdirSync(outDir, { recursive: true })

const target = join(outDir, 'clip.mp4')

execFileSync(
  process.env.FFMPEG_PATH ?? 'ffmpeg',
  [
    '-y',
    '-v', 'error',
    // Движущаяся мира: соседние кадры заметно различаются, поэтому и превью
    // ключевых кадров, и сегментация ведут себя как на настоящем ролике.
    '-f', 'lavfi',
    '-i', `testsrc2=size=1280x720:rate=30`,
    '-t', String(seconds),
    '-c:v', 'libvpx-vp9',
    '-b:v', '1M',
    '-pix_fmt', 'yuv420p',
    target,
  ],
  { stdio: 'inherit' },
)

console.log('готово:', target)
