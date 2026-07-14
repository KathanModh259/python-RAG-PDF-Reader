import { useNavigate } from 'react-router-dom';
import { useLanguage } from '@/lib/language';

export default function PanicButton() {
  const navigate = useNavigate();
  const { t } = useLanguage();

  return (
    <div className="fixed bottom-6 right-6 z-30 group">
      <button
        onClick={() => navigate('/panic')}
        className="h-16 w-16 rounded-full bg-gradient-to-br from-[#DC2626] to-[#EA580C] text-white shadow-lg hover:shadow-xl hover:scale-110 transition-all duration-300 pulse-gentle flex items-center justify-center"
        aria-label="Panic Mode - Emergency Help"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      </button>
      <div className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 hidden group-hover:block">
        <div className="bg-[#1F2937] text-white text-sm font-medium px-3 py-1.5 rounded-lg whitespace-nowrap shadow-lg">
          {t('Emergency', 'आपातकाल')}
        </div>
        <div className="w-2 h-2 bg-[#1F2937] rotate-45 absolute -bottom-1 left-1/2 -translate-x-1/2" />
      </div>
    </div>
  );
}
