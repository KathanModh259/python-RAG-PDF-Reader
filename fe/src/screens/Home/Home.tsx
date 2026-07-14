import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import HelplineSidebar from '@/components/HelplineSidebar';
import PanicButton from '@/components/PanicButton';
import { Button } from '@/components/ui/button';
import type { TileData } from '@/lib/types';
import { useLanguage } from '@/lib/language';

const tiles: TileData[] = [
  {
    id: 'notice',
    emoji: '📄',
    titleEn: 'I got a notice',
    titleHi: 'मुझे कोई नोटिस मिला है',
    bgClass: 'bg-blue-50 hover:bg-blue-100',
    route: '/guided-flow/notice',
  },
  {
    id: 'police',
    emoji: '👮',
    titleEn: 'Trouble with police',
    titleHi: 'पुलिस से परेशानी',
    bgClass: 'bg-amber-50 hover:bg-amber-100',
    route: '/guided-flow/police',
  },
  {
    id: 'landlord',
    emoji: '🏠',
    titleEn: 'Landlord or tenant problem',
    titleHi: 'मकान-मालिक / किरायेदार का झगड़ा',
    bgClass: 'bg-green-50 hover:bg-green-100',
    route: '/guided-flow/landlord',
  },
  {
    id: 'money',
    emoji: '💰',
    titleEn: 'Someone owes me money',
    titleHi: 'कोई मेरा पैसा नहीं लौटा रहा',
    bgClass: 'bg-purple-50 hover:bg-purple-100',
    route: '/guided-flow/money',
  },
  {
    id: 'online',
    emoji: '📱',
    titleEn: 'Online harassment',
    titleHi: 'ऑनलाइन परेशानी',
    bgClass: 'bg-pink-50 hover:bg-pink-100',
    route: '/guided-flow/online',
  },
  {
    id: 'domestic',
    emoji: '👩',
    titleEn: 'Domestic issue',
    titleHi: 'घरेलू समस्या',
    bgClass: 'bg-rose-50 hover:bg-rose-100',
    route: '/guided-flow/domestic',
  },
  {
    id: 'traffic',
    emoji: '🚗',
    titleEn: 'Traffic challan confusion',
    titleHi: 'ट्रैफिक चालान',
    bgClass: 'bg-cyan-50 hover:bg-cyan-100',
    route: '/guided-flow/traffic',
  },
  {
    id: 'rti',
    emoji: '🧾',
    titleEn: 'I want to file an RTI',
    titleHi: 'RTI दर्ज करें',
    bgClass: 'bg-orange-50 hover:bg-orange-100',
    route: '/guided-flow/rti',
  },
];

export default function Home() {
  const navigate = useNavigate();
  const { t, language, setLanguage } = useLanguage();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [helplineOpen, setHelplineOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);

  const recentCases = [
    { id: '1', title: t('Traffic Challan', 'ट्रैफिक चालान'), date: t('2 days ago', '2 दिन पहले'), risk: 'low' as const },
    { id: '2', title: t('Notice from Bank', 'बैंक से नोटिस'), date: t('1 week ago', '1 हफ्ते पहले'), risk: 'medium' as const },
  ];

  return (
    <div className="min-h-screen bg-[#FAFAF7]">
      {/* Top Bar */}
      <header className="sticky top-0 z-20 bg-white/80 backdrop-blur-lg border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors lg:hidden"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
            </button>
            <div
              onClick={() => navigate('/')}
              className="flex items-center gap-2 cursor-pointer"
            >
              <span className="text-2xl">⚖️</span>
              <span className="text-xl font-bold text-[#1F2937]">
                {t('Nyaya Mitra', 'न्याय मित्र')}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-gray-100 rounded-2xl p-0.5">
              <button
                onClick={() => setLanguage('en')}
                className={`px-3 py-1.5 rounded-xl text-sm font-medium transition-all ${
                  language === 'en' ? 'bg-white text-[#1F2937] shadow-sm' : 'text-[#6B7280]'
                }`}
              >
                EN
              </button>
              <button
                onClick={() => setLanguage('hi')}
                className={`px-3 py-1.5 rounded-xl text-sm font-medium transition-all ${
                  language === 'hi' ? 'bg-white text-[#1F2937] shadow-sm' : 'text-[#6B7280]'
                }`}
              >
                हि
              </button>
            </div>

            <button
              onClick={() => setHelplineOpen(true)}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors"
              title={t('Emergency Helplines', 'आपातकालीन हेल्पलाइन')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
            </button>

            <button
              onClick={() => navigate('/settings')}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors"
              title={t('Settings', 'सेटिंग्स')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            </button>

            <button
              onClick={() => navigate('/chat')}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors"
              title={t('Chat', 'चैट')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </button>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Left Sidebar - Recent Cases */}
        <aside
          className={`${
            sidebarCollapsed ? 'hidden' : 'block'
          } lg:block w-64 bg-white border-r border-gray-100 shrink-0 min-h-[calc(100vh-4rem)] overflow-y-auto transition-all`}
        >
          <div className="p-4">
            <h3 className="text-sm font-semibold text-[#6B7280] uppercase tracking-wider mb-4">
              {t('Recent Cases', 'हाल के मामले')}
            </h3>
            {recentCases.length === 0 ? (
              <p className="text-sm text-[#9CA3AF] text-center py-8">
                {t('No cases yet', 'अभी कोई मामला नहीं')}
              </p>
            ) : (
              <div className="space-y-2">
                {recentCases.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => navigate('/response')}
                    className="w-full text-left p-3 rounded-2xl hover:bg-[#FAFAF7] transition-colors border border-transparent hover:border-gray-100"
                  >
                    <div className="flex items-center gap-2">
                      <span className={`h-2 w-2 rounded-full shrink-0 ${
                        c.risk === 'low' ? 'bg-green-500' : c.risk === 'medium' ? 'bg-[#F59E0B]' : 'bg-[#DC2626]'
                      }`} />
                      <span className="text-sm font-medium text-[#1F2937] truncate">{c.title}</span>
                    </div>
                    <span className="text-xs text-[#9CA3AF] ml-4">{c.date}</span>
                  </button>
                ))}
                <button
                  onClick={() => navigate('/documents')}
                  className="w-full text-left p-3 rounded-2xl hover:bg-[#FAFAF7] transition-colors text-sm text-[#0F766E] font-medium"
                >
                  {t('View all cases →', 'सभी मामले देखें →')}
                </button>
              </div>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8">
          <div className="max-w-5xl mx-auto">
            <div className="mb-8">
              <h1 className="text-2xl sm:text-3xl font-bold text-[#1F2937] mb-2">
                {t('How can I help you?', 'मैं आपकी कैसे मदद कर सकता हूँ?')}
              </h1>
              <p className="text-[#6B7280]">
                {t('Pick a topic below or just start chatting', 'नीचे कोई विषय चुनें या बस चैट शुरू करें')}
              </p>
            </div>

            {/* 4x2 Tile Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {tiles.map((tile) => (
                <button
                  key={tile.id}
                  onClick={() => navigate(tile.route)}
                  className={`${tile.bgClass} p-6 rounded-2xl border border-transparent hover:border-gray-200 transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 text-left group`}
                >
                  <div className="text-3xl mb-3 group-hover:scale-110 transition-transform duration-200 inline-block">
                    {tile.emoji}
                  </div>
                  <div className="font-semibold text-[#1F2937] text-base leading-snug mb-1">
                    {tile.titleEn}
                  </div>
                  <div className="text-sm text-[#6B7280] leading-snug">
                    {tile.titleHi}
                  </div>
                </button>
              ))}
            </div>

            {/* Micro Tile */}
            <button
              onClick={() => navigate('/chat')}
              className="mt-4 w-full p-4 rounded-2xl bg-gradient-to-r from-[#0F766E]/5 to-[#F59E0B]/5 hover:from-[#0F766E]/10 hover:to-[#F59E0B]/10 border border-dashed border-gray-200 hover:border-[#0F766E]/30 transition-all duration-200 text-left flex items-center justify-between group"
            >
              <div>
                <span className="text-xl mr-3">💬</span>
                <span className="font-medium text-[#1F2937]">
                  {t('Something else? Just tell me →', 'कुछ और? बस मुझे बताएं →')}
                </span>
              </div>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[#0F766E] group-hover:translate-x-1 transition-transform"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
            </button>
          </div>
        </main>

        {/* Right Sidebar Trigger */}
        <button
          onClick={() => setHelplineOpen(true)}
          className="hidden lg:flex fixed right-0 top-1/2 -translate-y-1/2 z-10 bg-white border border-gray-200 border-r-0 rounded-l-2xl shadow-sm hover:shadow-md transition-all p-3 items-center gap-2 group"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[#0F766E]"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
          <span className="text-sm font-medium text-[#1F2937] hidden group-hover:inline">
            {t('Helplines', 'हेल्पलाइन')}
          </span>
        </button>
      </div>

      {/* Helpline Sidebar */}
      <HelplineSidebar isOpen={helplineOpen} onClose={() => setHelplineOpen(false)} language={language} />

      {/* Panic Mode Button */}
      <PanicButton />
    </div>
  );
}
