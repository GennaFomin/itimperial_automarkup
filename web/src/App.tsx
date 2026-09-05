import { HashRouter, Route, Routes } from 'react-router-dom'
import { TasksScreen } from './screens/TasksScreen'
import { EditorScreen } from './screens/EditorScreen'
import { Toaster } from './components/Toaster'

export function App() {
  // HashRouter: сборка открывается и как статические файлы, без настройки сервера.
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<TasksScreen />} />
        <Route path="/task/:taskId" element={<EditorScreen />} />
        <Route path="*" element={<TasksScreen />} />
      </Routes>
      <Toaster />
    </HashRouter>
  )
}
