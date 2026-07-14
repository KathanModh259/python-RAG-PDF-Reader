import { useState } from 'react';
import { cn } from '@/lib/utils';
import type { Helpline } from '@/lib/types';
import { useLanguage } from '@/lib/language';

const helplines: Helpline[] = [
  { id: 'women', name: 'Women Helpline', nameHi: 'महिला हेल्पलाइन', number: '181', description: 'For women in distress', descriptionHi: 'संकटग्रस्त महिलाओं के लिए' },
  { id: 'domestic', name: 'Domestic Violence', nameHi: 'घरेलू हिंसा', number: '181', description: 'Domestic violence support', descriptionHi: 'घरेलू हिंसा सहायता' },
  { id: 'kiran', name: 'Kiran Helpline', nameHi: 'किरण हेल्पलाइन', number: '1800-599-0019', description: 'Mental health support', descriptionHi: 'मानसिक स्वास्थ्य सहायता' },
  { id: 'icall', name: 'iCall', nameHi: 'आईकॉल', number: '9152987821', description: 'Mental health counselling', descriptionHi: 'मानसिक स्वास्थ्य परामर्श' },
  { id: 'child', name: 'Child Helpline', nameHi: 'बाल हेल्पलाइन', number: '1098', description: 'For children in need', descriptionHi: 'जरूरतमंद बच्चों के लिए' },
  { id: 'police', name: 'Police', nameHi: 'पुलिस', number: '112', description: 'Emergency police', descriptionHi: 'आपातकालीन पुलिस' },
  { id: 'cyber', name: 'Cyber Crime', nameHi: 'साइबर क्राइम', number: '1930', description: 'Cyber crime reporting', descriptionHi: 'साइबर अपराध रिपोर्टिंग' },
  { id: 'nalsa', name: 'NALSA', nameHi: 'नालसा', number: '15100', description: 'Free legal aid', descriptionHi: 'मुफ्त कानूनी सहायता' },
];

interface HelplineSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  language?: 'en' | 'hi';
}

export default function HelplineSidebar({ isOpen, onClose }: HelplineSidebarProps) {
  const { t, language } = useLanguage();
  const [copied, setCopied] = useState<string | null>(null);

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
    <>
      {isOpen && (
        <div className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm" onClick={onClose} />
      )}
      <div
        className={cn(
          'fixed top-0 right-0 z-50 h-full w-80 bg-white shadow-2xl border-l border-gray-200 transform transition-transform duration-300 ease-in-out overflow-y-auto',
          isOpen ? 'translate-x-0' : 'translate-x-full'
        )}
      >
        <div className="sticky top-0 bg-white border-b border-gray-100 px-5 py-4 flex items-center justify-between z-10">
          <h2 className="text-lg font-semibold text-[#1F2937]">
            {t('Emergency Helplines', 'आपातकालीन हेल्पलाइन')}
          </h2>
          <button
            onClick={onClose}
            className="h-8 w-8 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div className="p-4 space-y-3">
          <p className="text-sm text-[#6B7280] mb-2">
            {t('Tap a number to call or copy', 'त्वरित सहायता के लिए नंबर पर टैप करें')}
          </p>
          {helplines.map((h) => (
            <button
              key={h.id}
              onClick={() => copyNumber(h.number)}
              className="w-full text-left p-4 rounded-2xl bg-[#FAFAF7] hover:bg-[#F0F0EA] border border-gray-100 transition-all duration-200 hover:shadow-md group"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-[#1F2937] text-sm">
                    {t(h.name, h.nameHi)}
                  </div>
                  <div className="text-xs text-[#6B7280] mt-0.5">
                    {t(h.description, h.descriptionHi)}
                  </div>
                  {copied === h.number && (
                    <div className="text-xs text-green-600 mt-1 font-medium">
                      {t('Copied!', 'कॉपी हुआ!')}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold text-[#0F766E]">{h.number}</span>
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[#6B7280] opacity-0 group-hover:opacity-100 transition-opacity"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                </div>
              </div>
            </button>
          ))}
          <p className="text-xs text-center text-[#9CA3AF] mt-4">
            {t('All helplines are available 24x7', 'सभी हेल्पलाइन 24x7 उपलब्ध हैं')}
          </p>
        </div>
      </div>
    </>
  );
}
