import { HashRouter, Routes, Route } from 'react-router-dom';
import { LanguageProvider } from '@/lib/language';
import PanicMode from '@/screens/PanicMode/PanicMode';
import Response from '@/screens/Response/Response';
import GuidedFlow from '@/screens/GuidedFlow/GuidedFlow';
import Chat from '@/screens/Chat/Chat';
import DocumentLibrary from '@/screens/DocumentLibrary/DocumentLibrary';
import AntiExploitation from '@/screens/AntiExploitation/AntiExploitation';
import Settings from '@/screens/Settings/Settings';

function AppContent() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Chat />} />
        <Route path="/panic" element={<PanicMode />} />
        <Route path="/response" element={<Response />} />
        <Route path="/guided-flow/:flowId" element={<GuidedFlow />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/documents" element={<DocumentLibrary />} />
        <Route path="/anti-exploitation" element={<AntiExploitation />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </HashRouter>
  );
}

function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}

export default App;
