import { useState } from 'react';
import { Button } from '@/components/ui/button';

interface FirstRunWizardProps {
  onComplete: () => void;
}

const steps = [
  { id: 'welcome', label: 'Welcome' },
  { id: 'disclaimer', label: 'Disclaimer' },
  { id: 'model', label: 'Model' },
  { id: 'ready', label: 'Ready' },
];

export default function FirstRunWizard({ onComplete }: FirstRunWizardProps) {
  const [step, setStep] = useState(0);
  const [language, setLanguage] = useState<'en' | 'hi'>('en');
  const [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
  const [modelConsent, setModelConsent] = useState(false);

  const t = (en: string, hi: string) => language === 'hi' ? hi : en;

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(s => s + 1);
    } else {
      localStorage.setItem('nyaya-mitra-first-run', 'true');
      localStorage.setItem('nyaya-mitra-language', language);
      onComplete();
    }
  };

  const canProceed = () => {
    switch (step) {
      case 0: return true;
      case 1: return disclaimerAccepted;
      case 2: return modelConsent;
      case 3: return true;
      default: return false;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 via-white to-amber-50 flex items-center justify-center p-4">
      <div className="w-full max-w-lg mx-auto">
        {/* Progress dots */}
        <div className="flex justify-center gap-2 mb-12">
          {steps.map((s, i) => (
            <div
              key={s.id}
              className={`h-2.5 rounded-full transition-all duration-500 ${
                i === step ? 'w-10 bg-[#0F766E]' : i < step ? 'w-2.5 bg-[#0F766E]/40' : 'w-2.5 bg-gray-200'
              }`}
            />
          ))}
        </div>

        {/* Step content */}
        <div className="bg-white rounded-3xl shadow-xl border border-gray-100 p-8 md:p-10">
          {/* Step 0: Welcome */}
          {step === 0 && (
            <div className="text-center space-y-6">
              <div className="text-6xl">⚖️</div>
              <h1 className="text-3xl font-bold text-[#1F2937]">
                {t('Welcome to Nyaya Mitra', 'न्याय मित्र में आपका स्वागत है')}
              </h1>
              <p className="text-[#6B7280] text-lg leading-relaxed">
                {t(
                  'Your friendly legal companion. Get simple, clear help for everyday legal problems — in Hindi or English.',
                  'आपका साथी कानूनी सहायक। रोज़मर्रा की कानूनी समस्याओं के लिए सरल और स्पष्ट मदद — हिंदी या अंग्रेज़ी में।'
                )}
              </p>
              <div className="pt-4">
                <p className="text-sm font-medium text-[#6B7280] mb-3">
                  {t('Choose your language / अपनी भाषा चुनें', 'Choose your language / अपनी भाषा चुनें')}
                </p>
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={() => setLanguage('en')}
                    className={`px-8 py-3 rounded-2xl border-2 transition-all font-medium text-base ${
                      language === 'en'
                        ? 'border-[#0F766E] bg-[#0F766E]/5 text-[#0F766E]'
                        : 'border-gray-200 text-[#6B7280] hover:border-gray-300'
                    }`}
                  >
                    🇬🇧 English
                  </button>
                  <button
                    onClick={() => setLanguage('hi')}
                    className={`px-8 py-3 rounded-2xl border-2 transition-all font-medium text-base ${
                      language === 'hi'
                        ? 'border-[#0F766E] bg-[#0F766E]/5 text-[#0F766E]'
                        : 'border-gray-200 text-[#6B7280] hover:border-gray-300'
                    }`}
                  >
                    🇮🇳 हिंदी
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 1: Disclaimer */}
          {step === 1 && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-[#1F2937]">
                {t('One thing before we start', 'शुरू करने से पहले एक बात')}
              </h2>
              <div className="bg-[#FAFAF7] rounded-2xl p-5 border border-gray-100 space-y-4 text-[#4B5563] leading-relaxed">
                <p>
                  {t(
                    'Nyaya Mitra is an AI-powered legal assistant. It provides general legal information and guidance, NOT legal advice from a qualified lawyer.',
                    'न्याय मित्र एक AI-संचालित कानूनी सहायक है। यह सामान्य कानूनी जानकारी और मार्गदर्शन प्रदान करता है, किसी योग्य वकील की कानूनी सलाह नहीं।'
                  )}
                </p>
                <p>
                  {t(
                    'For serious legal matters, please consult a registered lawyer. Your data stays on your device.',
                    'गंभीर कानूनी मामलों के लिए कृपया एक पंजीकृत वकील से सलाह लें। आपका डेटा आपके डिवाइस पर ही रहता है।'
                  )}
                </p>
                <div className="flex items-start gap-3 pt-2">
                  <input
                    type="checkbox"
                    id="disclaimer"
                    checked={disclaimerAccepted}
                    onChange={(e) => setDisclaimerAccepted(e.target.checked)}
                    className="mt-1 h-5 w-5 rounded-lg border-gray-300 text-[#0F766E] focus:ring-[#0F766E] accent-[#0F766E]"
                  />
                  <label htmlFor="disclaimer" className="text-sm font-medium text-[#1F2937] cursor-pointer">
                    {t(
                      'I understand that this is not a substitute for a qualified lawyer.',
                      'मैं समझता/समझती हूँ कि यह किसी योग्य वकील का विकल्प नहीं है।'
                    )}
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Model Download */}
          {step === 2 && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-[#1F2937]">
                {t('One-time setup', 'एक बार का सेटअप')}
              </h2>
              <div className="bg-[#FAFAF7] rounded-2xl p-5 border border-gray-100 space-y-4">
                <div className="flex items-center gap-4">
                  <div className="h-12 w-12 rounded-2xl bg-[#0F766E]/10 flex items-center justify-center text-2xl">
                    🧠
                  </div>
                  <div>
                    <p className="font-medium text-[#1F2937]">
                      {t('Download legal AI model', 'कानूनी AI मॉडल डाउनलोड करें')}
                    </p>
                    <p className="text-sm text-[#6B7280]">
                      {t('About 2 GB • Takes 5-10 minutes', 'लगभग 2 GB • 5-10 मिनट लगते हैं')}
                    </p>
                  </div>
                </div>
                <p className="text-sm text-[#4B5563] leading-relaxed">
                  {t(
                    'This lets Nyaya Mitra work completely offline. Your data never leaves your computer.',
                    'इससे न्याय मित्र पूरी तरह ऑफलाइन काम कर सकता है। आपका डेटा आपके कंप्यूटर से कभी बाहर नहीं जाता।'
                  )}
                </p>
                <div className="flex items-start gap-3 pt-2">
                  <input
                    type="checkbox"
                    id="modelConsent"
                    checked={modelConsent}
                    onChange={(e) => setModelConsent(e.target.checked)}
                    className="mt-1 h-5 w-5 rounded-lg border-gray-300 text-[#0F766E] focus:ring-[#0F766E] accent-[#0F766E]"
                  />
                  <label htmlFor="modelConsent" className="text-sm font-medium text-[#1F2937] cursor-pointer">
                    {t(
                      'Yes, download the model. I have a stable internet connection.',
                      'हाँ, मॉडल डाउनलोड करें। मेरा इंटरनेट कनेक्शन स्थिर है।'
                    )}
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Ready */}
          {step === 3 && (
            <div className="text-center space-y-6">
              <div className="text-6xl">🎉</div>
              <h1 className="text-3xl font-bold text-[#1F2937]">
                {t('You\'re all set!', 'आप तैयार हैं!')}
              </h1>
              <p className="text-[#6B7280] text-lg leading-relaxed">
                {t(
                  'Start exploring Nyaya Mitra. Pick a topic or just describe what happened — we\'ll help you figure things out.',
                  'न्याय मित्र का उपयोग शुरू करें। कोई विषय चुनें या बस बताएं कि क्या हुआ — हम आपकी मदद करेंगे।'
                )}
              </p>
            </div>
          )}

          {/* Navigation */}
          <div className="mt-8 flex items-center justify-between">
            {step > 0 ? (
              <Button variant="outline" onClick={() => setStep(s => s - 1)}>
                {t('Back', 'पीछे')}
              </Button>
            ) : (
              <div />
            )}
            <Button onClick={handleNext} disabled={!canProceed()} size="lg">
              {step === steps.length - 1 ? t('Get Started', 'शुरू करें') : t('Continue', 'जारी रखें')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
