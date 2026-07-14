import { useState, useEffect, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '@/lib/language';
import { useBackend } from '@/hooks/useBackend';
import PanicButton from '@/components/PanicButton';
import { Button } from '@/components/ui/button';

const FONT_SIZE_KEY = 'nyaya-mitra-font-size';
const HIGH_CONTRAST_KEY = 'nyaya-mitra-high-contrast';
const VOICE_INPUT_KEY = 'nyaya-mitra-voice-input';
const READ_ALOUD_VOICE_KEY = 'nyaya-mitra-read-aloud-voice';

function loadFontSize(): number {
  try {
    return Number(localStorage.getItem(FONT_SIZE_KEY)) || 16;
  } catch {
    return 16;
  }
}

function loadHighContrast(): boolean {
  try {
    return localStorage.getItem(HIGH_CONTRAST_KEY) === 'true';
  } catch {
    return false;
  }
}

function loadVoiceInput(): boolean {
  try {
    return localStorage.getItem(VOICE_INPUT_KEY) !== 'false';
  } catch {
    return true;
  }
}

function loadReadAloudVoice(): string {
  try {
    return localStorage.getItem(READ_ALOUD_VOICE_KEY) || 'default';
  } catch {
    return 'default';
  }
}

function settingRow(
  label: ReactNode,
  description: ReactNode,
  control: ReactNode
) {
  return (
    <div className="flex items-center justify-between py-4 border-b border-gray-50 last:border-0">
      <div className="flex-1 pr-4">
        <div className="font-medium text-[#1F2937] text-sm">{label}</div>
        <div className="text-xs text-[#6B7280] mt-0.5">{description}</div>
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}

export default function Settings() {
  const navigate = useNavigate();
  const { t, language, setLanguage } = useLanguage();
  const { health } = useBackend();
  const [fontSize, setFontSize] = useState(loadFontSize);
  const [highContrast, setHighContrast] = useState(loadHighContrast);
  const [voiceInput, setVoiceInput] = useState(loadVoiceInput);
  const [readAloudVoice, setReadAloudVoice] = useState(loadReadAloudVoice);
  const [healthData, setHealthData] = useState<{ status: string; documents_indexed: number; model_loaded: boolean } | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [deleteConfirming, setDeleteConfirming] = useState(false);

  useEffect(() => {
    document.documentElement.style.fontSize = `${fontSize}px`;
    localStorage.setItem(FONT_SIZE_KEY, String(fontSize));
  }, [fontSize]);

  useEffect(() => {
    if (highContrast) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
    localStorage.setItem(HIGH_CONTRAST_KEY, String(highContrast));
  }, [highContrast]);

  useEffect(() => {
    localStorage.setItem(VOICE_INPUT_KEY, String(voiceInput));
  }, [voiceInput]);

  useEffect(() => {
    localStorage.setItem(READ_ALOUD_VOICE_KEY, readAloudVoice);
  }, [readAloudVoice]);

  useEffect(() => {
    let cancelled = false;
    health()
      .then((h) => { if (!cancelled) setHealthData(h); })
      .catch(() => { if (!cancelled) setHealthError(true); });
    return () => { cancelled = true; };
  }, [health]);

  const handleExportData = () => {
    const data: Record<string, string | null> = {};
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key) {
        data[key] = localStorage.getItem(key);
      }
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nyaya-mitra-backup-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDeleteAllData = () => {
    if (!deleteConfirming) {
      setDeleteConfirming(true);
      return;
    }
    const keysToKeep = ['nyaya-mitra-language'];
    const keys = Object.keys(localStorage);
    for (const key of keys) {
      if (!keysToKeep.includes(key)) {
        localStorage.removeItem(key);
      }
    }
    document.documentElement.style.fontSize = '';
    document.documentElement.classList.remove('high-contrast');
    window.location.reload();
  };

  const cancelDelete = () => setDeleteConfirming(false);

  return (
    <div className="min-h-screen bg-[#FAFAF7]">
      <header className="sticky top-0 z-20 bg-white/80 backdrop-blur-lg border-b border-gray-100">
        <div className="max-w-3xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6" /></svg>
            </button>
            <span className="text-xl font-bold text-[#1F2937]">
              {t('Settings', 'सेटिंग्स')}
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        {/* Language */}
        <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-base font-semibold text-[#1F2937] mb-4 flex items-center gap-2">
            <span className="text-xl">🌐</span>
            {t('Language', 'भाषा')}
          </h2>
          <div className="flex gap-3">
            <button
              onClick={() => setLanguage('en')}
              className={`flex-1 p-4 rounded-2xl border-2 text-center transition-all ${
                language === 'en'
                  ? 'border-[#0F766E] bg-[#0F766E]/5'
                  : 'border-gray-100 bg-[#FAFAF7] hover:border-gray-200'
              }`}
            >
              <div className="text-lg mb-1">🇬🇧</div>
              <div className="font-medium text-sm text-[#1F2937]">English</div>
            </button>
            <button
              onClick={() => setLanguage('hi')}
              className={`flex-1 p-4 rounded-2xl border-2 text-center transition-all ${
                language === 'hi'
                  ? 'border-[#0F766E] bg-[#0F766E]/5'
                  : 'border-gray-100 bg-[#FAFAF7] hover:border-gray-200'
              }`}
            >
              <div className="text-lg mb-1">🇮🇳</div>
              <div className="font-medium text-sm text-[#1F2937]">हिंदी</div>
            </button>
          </div>
        </div>

        {/* Appearance */}
        <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-base font-semibold text-[#1F2937] mb-4 flex items-center gap-2">
            <span className="text-xl">🎨</span>
            {t('Appearance', 'दिखावट')}
          </h2>
          <div className="divide-y divide-gray-50">
            {settingRow(
              <>{t('Font size', 'फ़ॉन्ट आकार')}</>,
              <>{t('Adjust text size for comfortable reading', 'आरामदायक पढ़ने के लिए टेक्स्ट आकार समायोजित करें')}</>,
              <div className="flex items-center gap-3">
                <span className="text-xs text-[#6B7280]">A</span>
                <input
                  type="range"
                  min="14"
                  max="24"
                  value={fontSize}
                  onChange={(e) => setFontSize(Number(e.target.value))}
                  className="w-24 h-2 rounded-full appearance-none cursor-pointer accent-[#0F766E] bg-gray-200"
                />
                <span className="text-lg text-[#1F2937] font-medium">A</span>
              </div>
            )}
            {settingRow(
              <>{t('High contrast mode', 'उच्च कंट्रास्ट मोड')}</>,
              <>{t('Increase contrast for better visibility', 'बेहतर दृश्यता के लिए कंट्रास्ट बढ़ाएं')}</>,
              <button
                onClick={() => setHighContrast(!highContrast)}
                className={`relative h-7 w-12 rounded-full transition-colors ${
                  highContrast ? 'bg-[#0F766E]' : 'bg-gray-200'
                }`}
              >
                <div className={`absolute top-0.5 h-6 w-6 rounded-full bg-white shadow-sm transition-transform ${
                  highContrast ? 'translate-x-5.5' : 'translate-x-0.5'
                }`} />
              </button>
            )}
          </div>
        </div>

        {/* Voice & Accessibility */}
        <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-base font-semibold text-[#1F2937] mb-4 flex items-center gap-2">
            <span className="text-xl">🎤</span>
            {t('Voice & Accessibility', 'आवाज़ और सुगमता')}
          </h2>
          <div className="divide-y divide-gray-50">
            {settingRow(
              <>{t('Voice input', 'आवाज़ इनपुट')}</>,
              <>{t('Use microphone to type with your voice', 'अपनी आवाज़ से टाइप करने के लिए माइक्रोफ़ोन का उपयोग करें')}</>,
              <button
                onClick={() => setVoiceInput(!voiceInput)}
                className={`relative h-7 w-12 rounded-full transition-colors ${
                  voiceInput ? 'bg-[#0F766E]' : 'bg-gray-200'
                }`}
              >
                <div className={`absolute top-0.5 h-6 w-6 rounded-full bg-white shadow-sm transition-transform ${
                  voiceInput ? 'translate-x-5.5' : 'translate-x-0.5'
                }`} />
              </button>
            )}
            {settingRow(
              <>{t('Read-aloud voice', 'पढ़कर सुनाने की आवाज़')}</>,
              <>{t('Choose voice for reading responses aloud', 'जवाब पढ़कर सुनाने के लिए आवाज़ चुनें')}</>,
              <select
                value={readAloudVoice}
                onChange={(e) => setReadAloudVoice(e.target.value)}
                className="px-3 py-2 rounded-xl border border-gray-200 bg-white text-sm text-[#1F2937] focus:outline-none focus:ring-2 focus:ring-[#0F766E]/20"
              >
                <option value="default">{t('Default', 'डिफ़ॉल्ट')}</option>
                <option value="male">{t('Male voice', 'पुरुष आवाज़')}</option>
                <option value="female">{t('Female voice', 'महिला आवाज़')}</option>
              </select>
            )}
          </div>
        </div>

        {/* AI Model Info */}
        <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-base font-semibold text-[#1F2937] mb-4 flex items-center gap-2">
            <span className="text-xl">🤖</span>
            {t('AI Model', 'AI मॉडल')}
          </h2>
          <div className="space-y-3 text-sm text-[#4B5563]">
            {healthError ? (
              <div className="text-red-500 text-xs">
                {t('Could not connect to backend. Make sure the server is running.', 'बैकएंड से कनेक्ट नहीं हो सका। कृपया सुनिश्चित करें कि सर्वर चल रहा है।')}
              </div>
            ) : healthData ? (
              <>
                <div className="flex items-center justify-between py-2 border-b border-gray-50">
                  <span>{t('Status', 'स्थिति')}</span>
                  <span className="text-[#0F766E] font-medium">{healthData.status}</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-gray-50">
                  <span>{t('Documents indexed', 'अनुक्रमित दस्तावेज़')}</span>
                  <span className="text-[#6B7280]">{healthData.documents_indexed}</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-gray-50">
                  <span>{t('Model loaded', 'मॉडल लोडेड')}</span>
                  <span className={`font-medium ${healthData.model_loaded ? 'text-[#0F766E]' : 'text-red-500'}`}>
                    {healthData.model_loaded ? t('Yes', 'हाँ') : t('No', 'नहीं')}
                  </span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span>{t('API URL', 'API URL')}</span>
                  <span className="text-[#6B7280] text-xs">http://127.0.0.1:8765</span>
                </div>
              </>
            ) : (
              <div className="flex items-center gap-2 text-[#6B7280] text-xs">
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                {t('Connecting to backend...', 'बैकएंड से कनेक्ट हो रहा है...')}
              </div>
            )}
          </div>
        </div>

        {/* Data & Storage */}
        <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-base font-semibold text-[#1F2937] mb-4 flex items-center gap-2">
            <span className="text-xl">💾</span>
            {t('Data & Storage', 'डेटा और स्टोरेज')}
          </h2>
          <div className="divide-y divide-gray-50">
            {settingRow(
              <>{t('Data location', 'डेटा स्थान')}</>,
              <>{t('All data is stored locally on your device', 'सारा डेटा आपके डिवाइस पर स्थानीय रूप से संग्रहीत है')}</>,
              <span className="text-xs text-[#6B7280] font-mono">LocalStorage</span>
            )}
          </div>
        </div>

        {/* About */}
        <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-base font-semibold text-[#1F2937] mb-4 flex items-center gap-2">
            <span className="text-xl">ℹ️</span>
            {t('About', 'बारे में')}
          </h2>
          <div className="space-y-3 text-sm text-[#4B5563]">
            <div className="flex items-center justify-between py-2 border-b border-gray-50">
              <span>{t('App version', 'ऐप वर्जन')}</span>
              <span className="text-[#6B7280]">1.0.0</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-gray-50">
              <span>{t('Data stored locally', 'डेटा स्थानीय रूप से संग्रहीत')}</span>
              <span className="text-[#6B7280]">{t('Yes', 'हाँ')}</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-gray-50">
              <span>{t('Open Source', 'ओपन सोर्स')}</span>
              <a
                href="https://github.com/anomalyco/nyaya-mitra"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#0F766E] hover:underline text-sm"
              >
                {t('View on GitHub', 'GitHub पर देखें')}
              </a>
            </div>
            <div className="flex items-center justify-between py-2">
              <span>{t('License', 'लाइसेंस')}</span>
              <span className="text-[#6B7280]">MIT</span>
            </div>
          </div>
        </div>

        {/* Data Actions */}
        <div className="bg-white rounded-3xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-base font-semibold text-[#1F2937] mb-4 flex items-center gap-2">
            <span className="text-xl">📁</span>
            {t('Data Management', 'डेटा प्रबंधन')}
          </h2>
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" size="sm" onClick={handleExportData}>
              {t('Export Data', 'डेटा निर्यात करें')}
            </Button>
            {deleteConfirming ? (
              <div className="flex items-center gap-2">
                <Button variant="destructive" size="sm" onClick={handleDeleteAllData}>
                  {t('Confirm Delete', 'हटाने की पुष्टि करें')}
                </Button>
                <Button variant="outline" size="sm" onClick={cancelDelete}>
                  {t('Cancel', 'रद्द करें')}
                </Button>
              </div>
            ) : (
              <Button
                variant="outline"
                size="sm"
                className="text-[#DC2626] border-[#DC2626]/30 hover:bg-red-50"
                onClick={handleDeleteAllData}
              >
                {t('Delete All Data', 'सारा डेटा हटाएं')}
              </Button>
            )}
          </div>
          {deleteConfirming && (
            <p className="text-xs text-red-600 mt-3">
              {t('This will permanently delete all your documents and data. This action cannot be undone.', 'यह आपके सभी दस्तावेज़ों और डेटा को स्थायी रूप से हटा देगा। यह क्रिया पूर्ववत नहीं की जा सकती।')}
            </p>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-[#9CA3AF] py-4">
          {t('Nyaya Mitra is not a substitute for a qualified lawyer. Always consult a professional for legal advice.', 'न्याय मित्र किसी योग्य वकील का विकल्प नहीं है। कानूनी सलाह के लिए हमेशा किसी पेशेवर से परामर्श करें।')}
        </p>
      </div>

      <PanicButton />
    </div>
  );
}
