import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '@/lib/language';
import PanicButton from '@/components/PanicButton';
import { Button } from '@/components/ui/button';
import type { DocumentInfo } from '@/lib/types';

const STORAGE_KEY = 'nyaya-mitra-documents';
const ANALYSES_KEY = 'nyaya-mitra-analyses-count';

type ViewMode = 'grid' | 'list';

const typeFilters = [
  { key: 'all', en: 'All', hi: 'सभी' },
  { key: 'notice', en: 'Notices', hi: 'नोटिस' },
  { key: 'court-order', en: 'Court Orders', hi: 'कोर्ट आदेश' },
  { key: 'agreement', en: 'Agreements', hi: 'समझौते' },
  { key: 'other', en: 'Other', hi: 'अन्य' },
] as const;

const typeBadgeColors: Record<string, string> = {
  notice: 'bg-blue-100 text-blue-700',
  'court-order': 'bg-purple-100 text-purple-700',
  agreement: 'bg-green-100 text-green-700',
  other: 'bg-gray-100 text-gray-600',
};

const typeLabels: Record<string, { en: string; hi: string }> = {
  notice: { en: 'Notice', hi: 'नोटिस' },
  'court-order': { en: 'Court Order', hi: 'कोर्ट आदेश' },
  agreement: { en: 'Agreement', hi: 'समझौता' },
  other: { en: 'Other', hi: 'अन्य' },
};

const typeEmojis: Record<string, string> = {
  notice: '📄',
  'court-order': '⚖️',
  agreement: '📝',
  other: '📁',
};

function loadDocuments(): DocumentInfo[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveDocuments(docs: DocumentInfo[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(docs));
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentLibrary() {
  const navigate = useNavigate();
  const { t, language } = useLanguage();
  const [docs, setDocs] = useState<DocumentInfo[]>(loadDocuments);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [analysesCount] = useState(() => {
    try {
      return Number(localStorage.getItem(ANALYSES_KEY)) || 0;
    } catch {
      return 0;
    }
  });

  const filtered = docs
    .filter((d) => {
      if (filterType !== 'all' && d.type !== filterType) return false;
      if (search && !d.name.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime());

  const toggleSelect = (id: string) => {
    setSelectedDocs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    if (selectedDocs.size === filtered.length) {
      setSelectedDocs(new Set());
    } else {
      setSelectedDocs(new Set(filtered.map((d) => d.id)));
    }
  };

  const handleDelete = (id: string) => {
    const doc = docs.find((d) => d.id === id);
    if (!doc) return;
    if (!window.confirm(t(`Delete "${doc.name}"?`, `"${doc.name}" हटाएं?`))) return;
    const updated = docs.filter((d) => d.id !== id);
    setDocs(updated);
    saveDocuments(updated);
    setSelectedDocs((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };

  const handleBulkDelete = () => {
    if (selectedDocs.size === 0) return;
    const count = selectedDocs.size;
    if (!window.confirm(t(`Delete ${count} selected document(s)?`, `${count} चयनित दस्तावेज़ हटाएं?`))) return;
    const updated = docs.filter((d) => !selectedDocs.has(d.id));
    setDocs(updated);
    saveDocuments(updated);
    setSelectedDocs(new Set());
  };

  const handleOpenFolder = (filePath?: string) => {
    if (filePath) {
      navigator.clipboard?.writeText(filePath);
    }
  };

  return (
    <div className="min-h-screen bg-[#FAFAF7]">
      <header className="sticky top-0 z-20 bg-white/80 backdrop-blur-lg border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="h-10 w-10 rounded-xl hover:bg-gray-100 flex items-center justify-center text-gray-500 transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6" /></svg>
            </button>
            <span className="text-xl font-bold text-[#1F2937]">
              {t('Document Library', 'दस्तावेज़ लाइब्रेरी')}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {selectedDocs.size > 0 && (
              <Button variant="destructive" size="sm" onClick={handleBulkDelete}>
                {t('Delete', 'हटाएं')} ({selectedDocs.size})
              </Button>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <div className="flex items-center gap-4 mb-6 text-sm text-[#6B7280]">
          <span>
            {language === 'hi'
              ? `${docs.length} दस्तावेज़`
              : `${docs.length} ${docs.length === 1 ? 'document' : 'documents'}`}
          </span>
          <span className="w-1 h-1 rounded-full bg-gray-300" />
          <span>
            {language === 'hi'
              ? `${analysesCount} कुल विश्लेषण`
              : `${analysesCount} total ${analysesCount === 1 ? 'analysis' : 'analyses'}`}
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 mb-4">
          <div className="relative flex-1">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="absolute left-4 top-1/2 -translate-y-1/2 text-[#9CA3AF]"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t('Search documents...', 'दस्तावेज़ खोजें...')}
              className="w-full pl-11 pr-4 py-2.5 rounded-2xl border border-gray-200 bg-white text-sm text-[#1F2937] placeholder:text-[#9CA3AF] focus:outline-none focus:ring-2 focus:ring-[#0F766E]/20 focus:border-[#0F766E]"
            />
          </div>
          <div className="flex bg-gray-100 rounded-2xl p-0.5">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-xl transition-all ${viewMode === 'grid' ? 'bg-white shadow-sm' : 'text-[#6B7280]'}`}
              title={t('Grid view', 'ग्रिड दृश्य')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /></svg>
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-xl transition-all ${viewMode === 'list' ? 'bg-white shadow-sm' : 'text-[#6B7280]'}`}
              title={t('List view', 'सूची दृश्य')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" /><line x1="3" y1="6" x2="3.01" y2="6" /><line x1="3" y1="12" x2="3.01" y2="12" /><line x1="3" y1="18" x2="3.01" y2="18" /></svg>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-6">
          {typeFilters.map(({ key, en, hi }) => (
            <button
              key={key}
              onClick={() => setFilterType(key)}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
                filterType === key
                  ? 'bg-[#0F766E] text-white'
                  : 'bg-white border border-gray-200 text-[#6B7280] hover:border-gray-300'
              }`}
            >
              {t(en, hi)}
            </button>
          ))}
        </div>

        {docs.length === 0 ? (
          <div className="text-center py-20">
            <div className="text-5xl mb-4">📂</div>
            <p className="text-[#6B7280] text-lg">
              {t('No documents yet', 'अभी तक कोई दस्तावेज़ नहीं')}
            </p>
            <p className="text-[#9CA3AF] text-sm mt-1">
              {t('Upload your first document to get started', 'शुरू करने के लिए अपना पहला दस्तावेज़ अपलोड करें')}
            </p>
            <Button variant="outline" className="mt-4" onClick={() => navigate('/guided-flow/notice')}>
              {t('Upload a document', 'दस्तावेज़ अपलोड करें')}
            </Button>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-20">
            <div className="text-5xl mb-4">🔍</div>
            <p className="text-[#6B7280] text-lg">
              {t('No matching documents', 'कोई मेल खाने वाला दस्तावेज़ नहीं')}
            </p>
            <Button variant="outline" className="mt-4" onClick={() => { setSearch(''); setFilterType('all'); }}>
              {t('Clear filters', 'फ़िल्टर साफ़ करें')}
            </Button>
          </div>
        ) : viewMode === 'grid' ? (
          <>
            {filtered.length > 1 && (
              <div className="flex items-center gap-2 mb-3">
                <label className="flex items-center gap-2 text-sm text-[#6B7280] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedDocs.size === filtered.length}
                    onChange={selectAll}
                    className="rounded border-gray-300 text-[#0F766E] focus:ring-[#0F766E]"
                  />
                  {t('Select all', 'सभी चुनें')}
                </label>
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filtered.map((doc) => (
                <div
                  key={doc.id}
                  className={`bg-white rounded-2xl border p-5 relative transition-all duration-200 hover:shadow-md ${
                    selectedDocs.has(doc.id) ? 'border-[#0F766E] ring-2 ring-[#0F766E]/20' : 'border-gray-100'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <button
                      onClick={() => navigate('/response', { state: { document: doc } })}
                      className="text-3xl hover:scale-110 transition-transform"
                    >
                      {typeEmojis[doc.type] || '📄'}
                    </button>
                    <label
                      onClick={(e) => e.stopPropagation()}
                      className={`h-5 w-5 rounded-md border-2 flex items-center justify-center cursor-pointer ${
                        selectedDocs.has(doc.id) ? 'bg-[#0F766E] border-[#0F766E]' : 'border-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedDocs.has(doc.id)}
                        onChange={() => toggleSelect(doc.id)}
                        className="sr-only"
                      />
                      {selectedDocs.has(doc.id) && (
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                      )}
                    </label>
                  </div>
                  <button
                    onClick={() => navigate('/response', { state: { document: doc } })}
                    className="text-left w-full"
                  >
                    <p className="font-medium text-[#1F2937] text-sm leading-snug line-clamp-2 mb-2">{doc.name}</p>
                    <div className="mb-2">
                      <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${typeBadgeColors[doc.type] || 'bg-gray-100 text-gray-600'}`}>
                        {t(typeLabels[doc.type]?.en || '', typeLabels[doc.type]?.hi || '')}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-[#6B7280]">
                      <span>{new Date(doc.uploadDate).toLocaleDateString()}</span>
                      <span>{doc.size != null ? formatSize(doc.size) : ''}</span>
                    </div>
                  </button>
                  <div className="flex gap-2 mt-3 pt-3 border-t border-gray-50">
                    {doc.filePath && (
                      <button
                        onClick={() => handleOpenFolder(doc.filePath)}
                        className="text-xs text-[#0F766E] hover:underline"
                        title={doc.filePath}
                      >
                        {t('Show path', 'पथ दिखाएं')}
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(doc.id)}
                      className="text-xs text-red-500 hover:underline ml-auto"
                    >
                      {t('Delete', 'हटाएं')}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
            {filtered.length > 1 && (
              <div className="px-5 py-3 border-b border-gray-50 flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-[#6B7280] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedDocs.size === filtered.length}
                    onChange={selectAll}
                    className="rounded border-gray-300 text-[#0F766E] focus:ring-[#0F766E]"
                  />
                  {t('Select all', 'सभी चुनें')}
                </label>
              </div>
            )}
            {filtered.map((doc) => (
              <div
                key={doc.id}
                className={`flex items-center gap-4 px-5 py-4 border-b border-gray-50 last:border-0 hover:bg-[#FAFAF7] transition-colors ${
                  selectedDocs.has(doc.id) ? 'bg-[#0F766E]/5' : ''
                }`}
              >
                <label className="shrink-0 cursor-pointer">
                  <div className={`h-5 w-5 rounded-md border-2 flex items-center justify-center ${
                    selectedDocs.has(doc.id) ? 'bg-[#0F766E] border-[#0F766E]' : 'border-gray-300'
                  }`}>
                    <input
                      type="checkbox"
                      checked={selectedDocs.has(doc.id)}
                      onChange={() => toggleSelect(doc.id)}
                      className="sr-only"
                    />
                    {selectedDocs.has(doc.id) && (
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                    )}
                  </div>
                </label>
                <button
                  onClick={() => navigate('/response', { state: { document: doc } })}
                  className="flex items-center gap-4 flex-1 min-w-0 text-left"
                >
                  <span className="text-xl shrink-0">
                    {typeEmojis[doc.type] || '📄'}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-[#1F2937] text-sm truncate">{doc.name}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${typeBadgeColors[doc.type] || 'bg-gray-100 text-gray-600'}`}>
                        {t(typeLabels[doc.type]?.en || '', typeLabels[doc.type]?.hi || '')}
                      </span>
                      <span className="text-xs text-[#9CA3AF]">{new Date(doc.uploadDate).toLocaleDateString()}</span>
                      {doc.size != null && (
                        <span className="text-xs text-[#9CA3AF]">{formatSize(doc.size)}</span>
                      )}
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => handleDelete(doc.id)}
                  className="text-xs text-red-500 hover:underline shrink-0"
                >
                  {t('Delete', 'हटाएं')}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <PanicButton />
    </div>
  );
}
