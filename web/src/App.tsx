import { useEffect, useState } from 'react'

import { Editor } from './Editor'
import { Library } from './Library'

/** Адрес вида #/video/<id>: перезагрузка не выбрасывает из редактора, ссылку можно переслать. */
const readHash = () => {
  const match = window.location.hash.match(/^#\/video\/([\w-]+)$/)
  return match ? match[1] : null
}

export default function App() {
  const [videoId, setVideoId] = useState<string | null>(readHash)

  useEffect(() => {
    const onHashChange = () => setVideoId(readHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const open = (id: string) => {
    window.location.hash = `#/video/${id}`
    setVideoId(id)
  }

  const back = () => {
    window.location.hash = ''
    setVideoId(null)
  }

  return videoId ? <Editor videoId={videoId} onBack={back} /> : <Library onOpen={open} />
}
