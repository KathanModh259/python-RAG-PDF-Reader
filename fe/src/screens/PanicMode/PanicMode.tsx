import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '@/lib/language';

const helplines = [
  { name: 'Women / Domestic Violence', nameHi: 'महिला / घरेलू हिंसा', number: '181' },
  { name: 'Kiran Mental Health', nameHi: 'किरण मानसिक स्वास्थ्य', number: '1800-599-0019' },
  { name: 'iCall Counselling', nameHi: 'आईकॉल परामर्श', number: '9152987821' },
  { name: 'Child Helpline', nameHi: 'बाल हेल्पलाइन', number: '1098' },
  { name: 'Police Emergency', nameHi: 'पुलिस आपातकाल', number: '112' },
  { name: 'Cyber Crime', nameHi: 'साइबर अपराध', number: '1930' },
  { name: 'NALSA Legal Aid', nameHi: 'नालसा कानूनी सहायता', number: '15100' },
];

const breathingPhases = [
  { text: 'Breathe in...', textHi: 'धीरे-धीरे साँस लें...', duration: 4000, scale: 1 },
  { text: 'Hold...', textHi: 'रोकें...', duration: 4000, scale: 1 },
  { text: 'Breathe out...', textHi: 'धीरे-धीरे छोड़ें...', duration: 4000, scale: 0.4 },
  { text: 'Hold...', textHi: 'रोकें...', duration: 4000, scale: 0.4 },
];

export default function PanicMode() {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [scale, setScale] = useState(0.4);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    let startTime = performance.now();
    let rafId: number;

    const animate = (timestamp: number) => {
      const elapsed = timestamp - startTime;
      const phase = breathingPhases[phaseIndex];
      const progress = Math.min(elapsed / phase.duration, 1);

      const eased = progress < 0.5 ? 2 * progress * progress : 1 - Math.pow(-2 * progress + 2, 2) / 2;

      const prevScale = phaseIndex === 0 ? 0.4 : breathingPhases[phaseIndex - 1].scale;
      const currentScale = prevScale + (phase.scale - prevScale) * eased;
      setScale(currentScale);

      if (progress >= 1) {
        setPhaseIndex((prev) => (prev + 1) % breathingPhases.length);
        startTime = timestamp;
      }

      rafId = requestAnimationFrame(animate);
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [phaseIndex]);

  const copyNumber = async (number: string) => {
    try {
      await navigator.clipboard.writeText(number);
      setCopied(number);
      setTimeout(() => setCopied(null), 2000);
    } catch {
      const textArea = document.createElement('textarea');
      textArea.value = number;
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(number);
      setTimeout(() => setCopied(null), 2000);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] bg-gradient-to-br from-teal-800 via-teal-700 to-cyan-800 flex flex-col items-center justify-center p-4 overflow-y-auto">
      {/* Breathing Circle */}
      <div className="flex-1 flex flex-col items-center justify-center min-h-0">
        <h1 className="text-2xl md:text-3xl font-bold text-white/90 mb-8 text-center">
          {t("You're safe. Take a breath.", 'आप सुरक्षित हैं। एक गहरी साँस लें।')}
        </h1>

        <div className="relative flex items-center justify-center mb-12">
          <div
            className="w-48 h-48 md:w-56 md:h-56 rounded-full bg-white/10 backdrop-blur-sm border-2 border-white/20 transition-all duration-300 flex items-center justify-center"
            style={{ transform: `scale(${scale})` }}
          >
            <span className="text-white/80 text-lg md:text-xl font-medium text-center px-4">
              {t(breathingPhases[phaseIndex].text, breathingPhases[phaseIndex].textHi)}
            </span>
          </div>
          {/* Outer rings */}
          <div
            className="absolute w-64 h-64 md:w-72 md:h-72 rounded-full border border-white/5 transition-all duration-300"
            style={{ transform: `scale(${scale * 1.1})` }}
          />
          <div
            className="absolute w-80 h-80 md:w-88 md:h-88 rounded-full border border-white/5 transition-all duration-300"
            style={{ transform: `scale(${scale * 1.2})` }}
          />
        </div>
      </div>

      {/* Helpline Buttons */}
      <div className="w-full max-w-2xl mx-auto">
        <p className="text-white/60 text-sm text-center mb-4">
          {t('Tap a number to copy it', 'नंबर कॉपी करने के लिए टैप करें')}
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {helplines.map((h) => (
            <button
              key={h.number}
              onClick={() => copyNumber(h.number)}
              className="bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/10 rounded-2xl p-4 text-white transition-all duration-200 hover:scale-105 active:scale-95"
            >
              <div className="text-lg font-semibold">{h.number}</div>
              <div className="text-xs text-white/70 mt-1 leading-tight">
                {t(h.name, h.nameHi)}
              </div>
              {copied === h.number && (
                <div className="text-xs text-green-300 mt-1">
                  {t('Copied!', 'कॉपी हुआ!')}
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Return button */}
      <button
        onClick={() => navigate(-1)}
        className="mt-8 mb-4 px-8 py-3 rounded-2xl bg-white/10 hover:bg-white/20 border border-white/10 text-white/80 hover:text-white transition-all duration-200 text-base font-medium"
      >
        {t("I'm safe now, take me back", 'मैं अब सुरक्षित हूँ, मुझे वापस ले चलें')}
      </button>
    </div>
  );
}
