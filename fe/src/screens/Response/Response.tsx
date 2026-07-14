import { useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useLanguage } from '@/lib/language';
import { Button } from '@/components/ui/button';
import PanicButton from '@/components/PanicButton';
import HelplineSidebar from '@/components/HelplineSidebar';

interface SourceCitation {
  text: string;
  heading: string;
  source: string;
  score: number;
}

interface ResponseState {
  answer: string;
  sources: SourceCitation[];
  confidence: number;
}

interface ParsedSection {
  header: string;
  content: string;
}

const KNOWN_SECTION_EMOJIS: Record<string, string> = {
  'what this is': '🔍',
  'what happened': '📋',
  'how serious is this': '⚠️',
  'what to do next': '🎯',
  'how to avoid being cheated': '🛡️',
  'how to prevent this in future': '🔒',
  'prevent': '🔒',
  'legal basis': '⚖️',
  'conclusion': '💡',
};

function parseAnswerIntoSections(answer: string): ParsedSection[] {
  const sections: ParsedSection[] = [];
  const lines = answer.split('\n');
  let currentHeader = '';
  let currentContent: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    const boldMatch = trimmed.match(/^\*\*(.+?)\*\*(?:\s*[:：]?\s*)?$/);
    const headingMatch = trimmed.match(/^#{1,3}\s+(.+)/);
    const numberedMatch = trimmed.match(/^\d+\.\s*\*\*(.+?)\*\*/);

    const header = boldMatch?.[1] || headingMatch?.[1] || numberedMatch?.[1];

    if (header) {
      if (currentHeader) {
        sections.push({ header: currentHeader, content: currentContent.join('\n').trim() });
      }
      currentHeader = header;
      currentContent = [];
    } else if (trimmed) {
      currentContent.push(line);
    } else {
      currentContent.push(line);
    }
  }

  if (currentHeader) {
    sections.push({ header: currentHeader, content: currentContent.join('\n').trim() });
  }

  if (sections.length === 0 && answer.trim()) {
    sections.push({ header: 'Analysis', content: answer.trim() });
  }

  return sections;
}

function getConfidenceInfo(confidence: number): { color: string; bg: string; labelEn: string; labelHi: string; dotColor: string } {
  if (confidence >= 0.7) {
    return {
      color: 'text-green-700',
      bg: 'bg-green-50',
      labelEn: 'High Confidence',
      labelHi: 'उच्च विश्वसनीयता',
      dotColor: 'bg-green-500',
    };
  }
  if (confidence >= 0.4) {
    return {
      color: 'text-amber-700',
      bg: 'bg-amber-50',
      labelEn: 'Medium Confidence',
      labelHi: 'मध्यम विश्वसनीयता',
      dotColor: 'bg-[#F59E0B]',
    };
  }
  return {
    color: 'text-red-700',
    bg: 'bg-red-50',
    labelEn: 'Low Confidence',
    labelHi: 'कम विश्वसनीयता',
    dotColor: 'bg-[#DC2626]',
  };
}

function getSourceIcon(heading: string): string {
  const lower = heading.toLowerCase();
  if (lower.includes('act') || lower.includes('section')) return '⚖️';
  if (lower.includes('court') || lower.includes('case')) return '🏛️';
  if (lower.includes('supreme') || lower.includes('high court')) return '🏛️';
  return '📄';
}

export default function Response() {
  const navigate = useNavigate();
  const location = useLocation();
  const { t, language } = useLanguage();

  const state = location.state as ResponseState | null;
  const hasData = state && state.answer && state.answer.trim().length > 0;

  const [expandedSections, setExpandedSections] = useState<Set<number>>(() => {
    if (hasData) return new Set([0]);
    return new Set();
  });
  const [helplineOpen, setHelplineOpen] = useState(false);
  const [explainMode, setExplainMode] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const sections: ParsedSection[] = hasData ? parseAnswerIntoSections(state!.answer) : [];
  const confidenceInfo = hasData ? getConfidenceInfo(state!.confidence) : null;

  const toggleSection = (index: number) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const handleSavePdf = () => {
    if (!hasData) return;
    const textContent = sections.map(s => `${s.header}\n${s.content}`).join('\n\n');
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(`
        <html>
          <head><title>Nyaya Mitra - Case Analysis</title></head>
          <body>
            <h1>Nyaya Mitra - Case Analysis</h1>
            <pre style="white-space: pre-wrap; font-family: sans-serif;">${textContent}</pre>
            <script>window.print();</script>
          </body>
        </html>
      `);
      printWindow.document.close();
    }
  };

  const handleShare = async () => {
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = url;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
  };

  const handleReadAloud = useCallback(() => {
    if (!hasData) return;
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const fullText = sections.map(s => `${s.header}. ${s.content}`).join('. ');
    const utterance = new SpeechSynthesisUtterance(fullText);
    utterance.lang = language === 'hi' ? 'hi-IN' : 'en-IN';
    utterance.rate = 0.9;
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
  }, [hasData, sections, language, isSpeaking]);

  const getSectionEmoji = (header: string): string => {
    const lower = header.toLowerCase().trim();
    for (const [key, emoji] of Object.entries(KNOWN_SECTION_EMOJIS)) {
      if (lower.includes(key)) return emoji;
    }
    return '📄';
  };

  if (!hasData) {
    return (
      <div className="min-h-screen bg-[#FAFAF7] flex items-center justify-center p-4">
        <div className="text-center max-w-md">
          <div className="text-6xl mb-6">📋</div>
          <h2 className="text-2xl font-bold text-[#1F2937] mb-3">
            {t('No Analysis Yet', 'अभी तक कोई विश्लेषण नहीं')}
          </h2>
          <p className="text-[#6B7280] mb-8">
            {t('No analysis yet. Start from the home screen.', 'अभी तक कोई विश्लेषण नहीं। होम स्क्रीन से शुरू करें।')}
          </p>
          <Button size="lg" onClick={() => navigate('/')}>
            {t('Go Home', 'होम पर जाएं')}
          </Button>
        </div>
        <PanicButton />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#FAFAF7] flex flex-col">
      {/* Top Bar */}
      <header className="sticky top-0 z-20 bg-white/80 backdrop-blur-lg border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6"/></svg>
            </button>
            <span className="text-xl font-bold text-[#1F2937]">
              {t('Case Analysis', 'केस विश्लेषण')}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* Confidence Badge */}
            {confidenceInfo && (
              <span className={`px-3 py-1.5 rounded-xl text-sm font-medium flex items-center gap-1.5 ${confidenceInfo.bg} ${confidenceInfo.color}`}>
                <span className={`h-2 w-2 rounded-full ${confidenceInfo.dotColor}`} />
                {t(confidenceInfo.labelEn, confidenceInfo.labelHi)}
              </span>
            )}

            {/* Save PDF */}
            <Button variant="ghost" size="sm" onClick={handleSavePdf}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              {t('Save PDF', 'PDF सहेजें')}
            </Button>

            {/* Share */}
            <Button variant="ghost" size="sm" onClick={handleShare}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
            </Button>

            {/* Explain toggle */}
            <Button
              variant={explainMode ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setExplainMode(!explainMode)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>
              {t('Explain', 'समझाएं')}
            </Button>

            {/* Read Aloud */}
            <Button
              variant={isSpeaking ? 'secondary' : 'ghost'}
              size="sm"
              onClick={handleReadAloud}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 8 14 2 18 6 17 10 13 14 9 20 6 22 4 19 10 15 14 10 18 6 14 2"/></svg>
              {isSpeaking ? t('Stop', 'रोकें') : t('Read Aloud', 'पढ़ें')}
            </Button>

            {/* Follow-up */}
            <Button variant="ghost" size="sm" onClick={() => navigate('/chat')}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              {t('Follow-up', 'आगे पूछें')}
            </Button>

            {/* Helpline */}
            <button
              onClick={() => setHelplineOpen(true)}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
            </button>
          </div>
        </div>
      </header>

      {/* 3-Column Layout */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left: Document Viewer */}
        <aside className="lg:w-[30%] bg-white border-b lg:border-b-0 lg:border-r border-gray-100 p-4 lg:overflow-y-auto">
          <h3 className="text-sm font-semibold text-[#6B7280] uppercase tracking-wider mb-4">
            {t('Source Documents', 'स्रोत दस्तावेज़')}
          </h3>
          <div className="rounded-2xl bg-[#FAFAF7] border border-gray-100 p-6 text-center">
            <div className="text-5xl mb-3">📄</div>
            <p className="text-sm text-[#6B7280]">
              {t('Uploaded documents analyzed', 'अपलोड किए गए दस्तावेज़ों का विश्लेषण किया गया')}
            </p>
            {state.sources && state.sources.length > 0 && (
              <p className="text-xs text-[#9CA3AF] mt-2">
                {t(
                  `${state.sources.length} source${state.sources.length !== 1 ? 's' : ''} cited`,
                  `${state.sources.length} स्रोत${state.sources.length !== 1 ? 'ों' : ''} का उल्लेख`
                )}
              </p>
            )}
          </div>
          {state.sources && state.sources.length > 0 && (
            <div className="mt-4 space-y-2">
              {state.sources.slice(0, 3).map((src, i) => (
                <div key={i} className="p-3 rounded-xl bg-[#FAFAF7] border border-gray-100 text-sm">
                  <p className="font-medium text-[#1F2937] truncate">{src.source}</p>
                  <p className="text-xs text-[#6B7280] mt-1 truncate">{src.heading}</p>
                </div>
              ))}
              {state.sources.length > 3 && (
                <p className="text-xs text-[#6B7280] text-center mt-1">
                  {t(`+${state.sources.length - 3} more`, `+${state.sources.length - 3} और`)}
                </p>
              )}
            </div>
          )}
        </aside>

        {/* Middle: Parsed sections */}
        <main className="flex-1 lg:w-[45%] overflow-y-auto p-4 lg:p-6 space-y-3">
          {sections.length === 0 ? (
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 text-center">
              <p className="text-[#6B7280]">{t('No structured analysis available.', 'कोई संरचित विश्लेषण उपलब्ध नहीं है।')}</p>
            </div>
          ) : (
            sections.map((section, index) => {
              const isExpanded = expandedSections.has(index);
              const emoji = getSectionEmoji(section.header);

              return (
                <div
                  key={index}
                  className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden transition-all duration-200 hover:shadow-md"
                >
                  <button
                    onClick={() => toggleSection(index)}
                    className="w-full p-4 flex items-center justify-between text-left"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-xl">{emoji}</span>
                      <span className="font-semibold text-[#1F2937]">
                        {section.header}
                      </span>
                    </div>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className={`text-[#6B7280] transition-transform duration-200 ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                    >
                      <polyline points="6 9 12 15 18 9" />
                    </svg>
                  </button>
                  {isExpanded && (
                    <div className="px-4 pb-4 pt-0 border-t border-gray-50">
                      <div className="text-[#4B5563] leading-relaxed whitespace-pre-line pt-3 text-sm">
                        {section.content}
                      </div>
                      {explainMode && (
                        <div className="mt-3 p-3 rounded-xl bg-[#0F766E]/5 border border-[#0F766E]/10 text-sm text-[#0F766E]">
                          💡 {t(
                            'Simplified: This explains the legal concept in everyday language.',
                            'सरल: यह कानूनी अवधारणा को सामान्य भाषा में समझाता है।'
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </main>

        {/* Right: Citations Panel */}
        <aside className="lg:w-[25%] bg-white border-t lg:border-t-0 lg:border-l border-gray-100 p-4 lg:overflow-y-auto">
          <h3 className="text-sm font-semibold text-[#6B7280] uppercase tracking-wider mb-4">
            {t('Citations', 'उद्धरण')}
          </h3>
          {state.sources && state.sources.length > 0 ? (
            <div className="space-y-3">
              {state.sources.map((src, i) => (
                <div key={i} className="p-3 rounded-2xl bg-[#FAFAF7] border border-gray-100">
                  <div className="flex items-center gap-2 mb-1">
                    <span>{getSourceIcon(src.heading)}</span>
                    <span className="text-xs text-[#6B7280] font-medium">{src.source}</span>
                  </div>
                  <p className="text-sm text-[#1F2937]">{src.heading}</p>
                  {src.text && (
                    <p className="text-xs text-[#6B7280] mt-1 line-clamp-3">{src.text}</p>
                  )}
                  <div className="flex items-center gap-1.5 mt-2">
                    <div className={`h-1.5 w-1.5 rounded-full ${
                      src.score >= 0.7 ? 'bg-green-500' : src.score >= 0.4 ? 'bg-[#F59E0B]' : 'bg-[#DC2626]'
                    }`} />
                    <span className="text-xs text-[#9CA3AF]">
                      {Math.round(src.score * 100)}% {t('relevance', 'प्रासंगिकता')}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-2xl bg-[#FAFAF7] border border-gray-100 p-6 text-center">
              <div className="text-3xl mb-2">📚</div>
              <p className="text-sm text-[#6B7280]">
                {t('No citations available', 'कोई उद्धरण उपलब्ध नहीं')}
              </p>
            </div>
          )}
        </aside>
      </div>

      <HelplineSidebar isOpen={helplineOpen} onClose={() => setHelplineOpen(false)} language={language} />
      <PanicButton />
    </div>
  );
}
