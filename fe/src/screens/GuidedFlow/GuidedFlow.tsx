import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useLanguage } from '@/lib/language';
import { useBackend } from '@/hooks/useBackend';
import { Button } from '@/components/ui/button';
import PanicButton from '@/components/PanicButton';

interface FlowConfig {
  steps: Array<{ id: string; skippable: boolean }>;
  step2Chips: Array<{ value: string; labelEn: string; labelHi: string }>;
  step2Question: { en: string; hi: string };
  step1Label: { en: string; hi: string };
}

const flowConfigs: Record<string, FlowConfig> = {
  notice: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'sender', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload the notice document', hi: 'नोटिस दस्तावेज़ अपलोड करें' },
    step2Question: { en: 'Who sent this notice?', hi: 'यह नोटिस किसने भेजा?' },
    step2Chips: [
      { value: 'bank', labelEn: 'Bank', labelHi: 'बैंक' },
      { value: 'court', labelEn: 'Court', labelHi: 'कोर्ट' },
      { value: 'police', labelEn: 'Police', labelHi: 'पुलिस' },
      { value: 'company', labelEn: 'Company', labelHi: 'कंपनी' },
      { value: 'individual', labelEn: 'Individual', labelHi: 'व्यक्ति' },
      { value: 'unsure', labelEn: 'Not sure', labelHi: 'पता नहीं' },
    ],
  },
  police: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'situation', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload any document (if available)', hi: 'कोई दस्तावेज़ अपलोड करें (यदि उपलब्ध हो)' },
    step2Question: { en: 'What happened?', hi: 'क्या हुआ?' },
    step2Chips: [
      { value: 'arrest', labelEn: 'Arrest or detention', labelHi: 'गिरफ्तारी या हिरासत' },
      { value: 'threat', labelEn: 'Threat by police', labelHi: 'पुलिस द्वारा धमकी' },
      { value: 'false_case', labelEn: 'False case registered', labelHi: 'झूठा मामला दर्ज' },
      { value: 'harassment', labelEn: 'Harassment', labelHi: 'उत्पीड़न' },
      { value: 'other', labelEn: 'Something else', labelHi: 'कुछ और' },
    ],
  },
  landlord: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'situation', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload rental agreement or notice', hi: 'किराया समझौता या नोटिस अपलोड करें' },
    step2Question: { en: 'What is the issue?', hi: 'समस्या क्या है?' },
    step2Chips: [
      { value: 'eviction', labelEn: 'Eviction threat', labelHi: 'बेदखली की धमकी' },
      { value: 'deposit', labelEn: 'Deposit refund', labelHi: 'जमा राशि वापसी' },
      { value: 'maintenance', labelEn: 'Maintenance issues', labelHi: 'रखरखाव की समस्याएं' },
      { value: 'harassment', labelEn: 'Harassment', labelHi: 'उत्पीड़न' },
      { value: 'other', labelEn: 'Something else', labelHi: 'कुछ और' },
    ],
  },
  money: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'situation', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload any proof document', hi: 'कोई सबूत दस्तावेज़ अपलोड करें' },
    step2Question: { en: 'Who owes you money?', hi: 'कौन आपका पैसा नहीं लौटा रहा?' },
    step2Chips: [
      { value: 'friend', labelEn: 'Friend', labelHi: 'दोस्त' },
      { value: 'employer', labelEn: 'Employer', labelHi: 'नियोक्ता' },
      { value: 'business', labelEn: 'Business partner', labelHi: 'व्यावसायिक भागीदार' },
      { value: 'relative', labelEn: 'Relative', labelHi: 'रिश्तेदार' },
      { value: 'other', labelEn: 'Someone else', labelHi: 'कोई और' },
    ],
  },
  online: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'situation', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload screenshots (if any)', hi: 'स्क्रीनशॉट अपलोड करें (यदि हों)' },
    step2Question: { en: 'What kind of online issue?', hi: 'किस तरह की ऑनलाइन समस्या?' },
    step2Chips: [
      { value: 'social', labelEn: 'Social media harassment', labelHi: 'सोशल मीडिया पर परेशानी' },
      { value: 'cyberbullying', labelEn: 'Cyberbullying', labelHi: 'साइबर बुलिंग' },
      { value: 'doxxing', labelEn: 'Doxxing / privacy breach', labelHi: 'निजता उल्लंघन' },
      { value: 'impersonation', labelEn: 'Fake account / impersonation', labelHi: 'नकली खाता' },
      { value: 'other', labelEn: 'Something else', labelHi: 'कुछ और' },
    ],
  },
  domestic: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'situation', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload any related document', hi: 'कोई संबंधित दस्तावेज़ अपलोड करें' },
    step2Question: { en: 'What kind of domestic issue?', hi: 'किस तरह की घरेलू समस्या?' },
    step2Chips: [
      { value: 'violence', labelEn: 'Domestic violence', labelHi: 'घरेलू हिंसा' },
      { value: 'financial', labelEn: 'Financial control', labelHi: 'आर्थिक नियंत्रण' },
      { value: 'mental', labelEn: 'Mental / emotional abuse', labelHi: 'मानसिक शोषण' },
      { value: 'custody', labelEn: 'Child custody', labelHi: 'बाल हिरासत' },
      { value: 'other', labelEn: 'Something else', labelHi: 'कुछ और' },
    ],
  },
  traffic: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'situation', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload challan or document', hi: 'चालान या दस्तावेज़ अपलोड करें' },
    step2Question: { en: 'What happened?', hi: 'क्या हुआ?' },
    step2Chips: [
      { value: 'parking', labelEn: 'Wrong parking challan', labelHi: 'गलत पार्किंग चालान' },
      { value: 'signal', labelEn: 'Signal jump challan', labelHi: 'सिग्नल जंप चालान' },
      { value: 'license', labelEn: 'Driving license issue', labelHi: 'ड्राइविंग लाइसेंस समस्या' },
      { value: 'metro', labelEn: 'Metro / e-challan confusion', labelHi: 'मेट्रो / ई-चालान' },
      { value: 'other', labelEn: 'Something else', labelHi: 'कुछ और' },
    ],
  },
  rti: {
    steps: [
      { id: 'upload', skippable: true },
      { id: 'situation', skippable: true },
      { id: 'urgency', skippable: true },
      { id: 'processing', skippable: false },
    ],
    step1Label: { en: 'Upload any reference document', hi: 'कोई संदर्भ दस्तावेज़ अपलोड करें' },
    step2Question: { en: 'What information do you need?', hi: 'आपको कौन सी जानकारी चाहिए?' },
    step2Chips: [
      { value: 'govt', labelEn: 'Government scheme info', labelHi: 'सरकारी योजना की जानकारी' },
      { value: 'police', labelEn: 'Police report / FIR', labelHi: 'पुलिस रिपोर्ट / FIR' },
      { value: 'municipal', labelEn: 'Municipal / civic info', labelHi: 'नगर निगम की जानकारी' },
      { value: 'personal', labelEn: 'Personal records', labelHi: 'व्यक्तिगत रिकॉर्ड' },
      { value: 'other', labelEn: 'Something else', labelHi: 'कुछ और' },
    ],
  },
};

const flowNames: Record<string, { en: string; hi: string }> = {
  notice: { en: 'I got a notice', hi: 'मुझे कोई नोटिस मिला है' },
  police: { en: 'Trouble with police', hi: 'पुलिस से परेशानी' },
  landlord: { en: 'Landlord / Tenant problem', hi: 'मकान-मालिक / किरायेदार का झगड़ा' },
  money: { en: 'Someone owes me money', hi: 'कोई मेरा पैसा नहीं लौटा रहा' },
  online: { en: 'Online harassment', hi: 'ऑनलाइन परेशानी' },
  domestic: { en: 'Domestic issue', hi: 'घरेलू समस्या' },
  traffic: { en: 'Traffic challan confusion', hi: 'ट्रैफिक चालान' },
  rti: { en: 'I want to file an RTI', hi: 'RTI दर्ज करें' },
};

export default function GuidedFlow() {
  const navigate = useNavigate();
  const { flowId = 'notice' } = useParams<{ flowId: string }>();
  const { t } = useLanguage();

  const config = flowConfigs[flowId] || flowConfigs.notice;
  const steps = config.steps;
  const flowName = flowNames[flowId] || flowNames.notice;

  const { upload, query } = useBackend();

  const [currentStep, setCurrentStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [chipsValue, setChipsValue] = useState<string | null>(null);
  const [urgency, setUrgency] = useState(3);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [processingStatus, setProcessingStatus] = useState<'idle' | 'uploading' | 'querying' | 'done' | 'error'>('idle');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStatusText, setProcessingStatusText] = useState('');
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [resultData, setResultData] = useState<{
    answer: string;
    sources: Array<{ text: string; heading: string; source: string; score: number }>;
    confidence: number;
  } | null>(null);

  const processingStepIndex = steps.findIndex(s => s.id === 'processing');

  const buildPrompt = useCallback(() => {
    const flowLabel = t(flowName.en, flowName.hi);
    const situation = chipsValue
      ? t(
          config.step2Chips.find(c => c.value === chipsValue)?.labelEn || '',
          config.step2Chips.find(c => c.value === chipsValue)?.labelHi || ''
        )
      : 'Not specified';
    const fileInfo = file
      ? `\n\nThe user has uploaded a document named: "${file.name}". Analyze its contents as part of the response.`
      : '';

    return `I am using the Nyaya Mitra legal assistant for the following situation:

**Flow:** ${flowLabel}
**Issue details:** ${situation}
**Urgency level (1-5):** ${urgency}${fileInfo}

Please provide a comprehensive legal analysis with the following 8 sections:
1. **What this is** — Explain what this legal situation is about in simple terms.
2. **What happened** — Summarize what has occurred based on the information provided.
3. **How serious is this?** — Assess the seriousness level (low/medium/high).
4. **What to do next** — Provide clear, actionable steps the person should take.
5. **How to avoid being cheated** — Warn about common scams or exploitation tactics related to this situation.
6. **How to prevent this in future** — Practical preventive measures.
7. **Legal basis** — Relevant Indian laws, acts, or sections that apply.
8. **Conclusion** — Final summary and recommendations.

Write the response in clear, simple language. Use the same language as this query. Format each section with a bold header like **What this is**.`;
  }, [flowName, chipsValue, urgency, file, config, t]);

  const startProcessing = useCallback(async () => {
    setProcessingError(null);
    setProcessingProgress(0);
    setProcessingStatus('idle');

    try {
      if (file) {
        setProcessingStatus('uploading');
        setProcessingStatusText(t('Uploading your document...', 'आपका दस्तावेज़ अपलोड कर रहे हैं...'));
        setProcessingProgress(10);

        await upload(file);

        setProcessingProgress(30);
        setProcessingStatusText(t('Document processed. Analyzing...', 'दस्तावेज़ संसाधित हुआ। विश्लेषण कर रहे हैं...'));
      } else {
        setProcessingProgress(30);
      }

      setProcessingStatus('querying');
      setProcessingProgress(40);
      setProcessingStatusText(t('Reading your document...', 'आपका दस्तावेज़ पढ़ रहे हैं...'));

      const prompt = buildPrompt();

      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => (prev < 85 ? prev + 3 : prev));
      }, 800);

      let result;
      try {
        result = await query(prompt, 'standard');
      } finally {
        clearInterval(progressInterval);
      }

      setProcessingProgress(90);
      setProcessingStatusText(t('Finding relevant laws...', 'प्रासंगिक कानून खोज रहे हैं...'));
      await new Promise(r => setTimeout(r, 300));

      setProcessingProgress(95);
      setProcessingStatusText(t('Preparing your response...', 'आपका उत्तर तैयार कर रहे हैं...'));
      await new Promise(r => setTimeout(r, 300));

      setProcessingProgress(100);
      setProcessingStatusText(t('Done!', 'तैयार!'));
      setProcessingStatus('done');
      setResultData(result);
    } catch (err) {
      setProcessingStatus('error');
      setProcessingError(err instanceof Error ? err.message : t('An error occurred', 'एक त्रुटि हुई'));
    }
  }, [file, upload, buildPrompt, query, t]);

  useEffect(() => {
    if (currentStep === processingStepIndex && processingStatus === 'idle') {
      startProcessing();
    }
  }, [currentStep, processingStepIndex, processingStatus, startProcessing]);

  const handleViewResults = () => {
    if (resultData) {
      navigate('/response', {
        state: {
          answer: resultData.answer,
          sources: resultData.sources,
          confidence: resultData.confidence,
        },
      });
    }
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(s => s + 1);
    } else {
      navigate('/response');
    }
  };

  const handleSkip = () => {
    const nextUnskippable = steps.findIndex((s, i) => i > currentStep && !s.skippable);
    if (nextUnskippable !== -1) {
      setCurrentStep(nextUnskippable);
    } else if (currentStep < steps.length - 1) {
      setCurrentStep(s => s + 1);
    } else {
      navigate('/response');
    }
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) setFile(files[0]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) setFile(files[0]);
  };

  const renderStepContent = () => {
    const step = steps[currentStep];
    if (!step) return null;

    switch (step.id) {
      case 'upload':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-[#1F2937]">
              {t(config.step1Label.en, config.step1Label.hi)}
            </h3>
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200 ${
                isDragging
                  ? 'border-[#0F766E] bg-[#0F766E]/5'
                  : file
                  ? 'border-green-300 bg-green-50'
                  : 'border-gray-200 hover:border-[#0F766E]/30 hover:bg-gray-50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.jpg,.jpeg,.png,.doc,.docx"
                onChange={handleFileSelect}
                className="hidden"
              />
              {file ? (
                <div>
                  <div className="text-4xl mb-3">📄</div>
                  <p className="font-medium text-[#1F2937]">{file.name}</p>
                  <p className="text-sm text-[#6B7280] mt-1">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  <Button variant="outline" size="sm" className="mt-3" onClick={(e) => { e.stopPropagation(); setFile(null); }}>
                    {t('Remove', 'हटाएं')}
                  </Button>
                </div>
              ) : (
                <div>
                  <div className="text-4xl mb-3">📤</div>
                  <p className="font-medium text-[#1F2937]">
                    {t('Drop your document here', 'अपना दस्तावेज़ यहाँ डालें')}
                  </p>
                  <p className="text-sm text-[#6B7280] mt-1">
                    {t('or click to browse', 'या ब्राउज़ करने के लिए क्लिक करें')}
                  </p>
                  <p className="text-xs text-[#9CA3AF] mt-3">
                    {t('PDF, JPG, PNG — up to 10 MB', 'PDF, JPG, PNG — 10 MB तक')}
                  </p>
                </div>
              )}
            </div>
          </div>
        );

      case 'sender':
      case 'situation':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-[#1F2937]">
              {t(config.step2Question.en, config.step2Question.hi)}
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {config.step2Chips.map((chip) => (
                <button
                  key={chip.value}
                  onClick={() => setChipsValue(chip.value)}
                  className={`p-4 rounded-2xl border-2 text-left transition-all duration-200 ${
                    chipsValue === chip.value
                      ? 'border-[#0F766E] bg-[#0F766E]/5'
                      : 'border-gray-100 bg-[#FAFAF7] hover:border-gray-200'
                  }`}
                >
                  <div className="font-medium text-[#1F2937]">{t(chip.labelEn, chip.labelHi)}</div>
                </button>
              ))}
            </div>
          </div>
        );

      case 'urgency':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-[#1F2937]">
              {t('How urgent is this?', 'यह कितना जरूरी है?')}
            </h3>
            <div className="px-2">
              <input
                type="range"
                min="1"
                max="5"
                value={urgency}
                onChange={(e) => setUrgency(Number(e.target.value))}
                className="w-full h-2 rounded-full appearance-none cursor-pointer accent-[#0F766E] bg-gray-200 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-6 [&::-webkit-slider-thumb]:w-6 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#0F766E] [&::-webkit-slider-thumb]:shadow-md"
              />
              <div className="flex justify-between text-sm text-[#6B7280] mt-2">
                <span>{t('Low', 'कम')}</span>
                <span>{t('High', 'उच्च')}</span>
              </div>
            </div>
            <div className="text-center">
              <div className={`inline-block px-4 py-2 rounded-2xl text-sm font-medium ${
                urgency <= 2 ? 'bg-green-50 text-green-700' :
                urgency <= 3 ? 'bg-amber-50 text-amber-700' :
                'bg-red-50 text-red-700'
              }`}>
                {urgency <= 2 ? t('Low urgency', 'कम जरूरी') :
                 urgency <= 3 ? t('Medium urgency', 'मध्यम जरूरी') :
                 t('High urgency', 'बहुत जरूरी')}
              </div>
            </div>
          </div>
        );

      case 'processing':
        return (
          <div className="space-y-6 text-center">
            {processingStatus === 'error' ? (
              <>
                <div className="text-5xl mb-4">❌</div>
                <h3 className="text-lg font-semibold text-[#DC2626]">
                  {t('Something went wrong', 'कुछ गलत हो गया')}
                </h3>
                <p className="text-[#6B7280] text-sm">{processingError}</p>
                <div className="flex gap-3 justify-center">
                  <Button variant="outline" onClick={() => navigate('/')}>
                    {t('Go Back', 'वापस जाएं')}
                  </Button>
                  <Button onClick={startProcessing}>
                    {t('Try Again', 'पुनः प्रयास करें')}
                  </Button>
                </div>
              </>
            ) : processingStatus === 'done' ? (
              <>
                <div className="text-5xl mb-4">✅</div>
                <h3 className="text-lg font-semibold text-[#1F2937]">
                  {t('Analysis complete!', 'विश्लेषण पूर्ण!')}
                </h3>
                <Button size="lg" onClick={handleViewResults}>
                  {t('See Results', 'परिणाम देखें')}
                </Button>
              </>
            ) : (
              <>
                <div className="text-5xl mb-4 animate-spin">⚖️</div>
                <h3 className="text-lg font-semibold text-[#1F2937]">
                  {t('Analyzing your case...', 'आपके केस का विश्लेषण कर रहे हैं...')}
                </h3>
                <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-[#0F766E] to-[#F59E0B] rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${processingProgress}%` }}
                  />
                </div>
                <p className="text-[#6B7280] text-sm">{processingStatusText}</p>
                <Button variant="outline" size="sm" onClick={() => navigate('/')}>
                  {t('Cancel', 'रद्द करें')}
                </Button>
              </>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  const currentStepObj = steps[currentStep];
  if (!currentStepObj) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 via-white to-amber-50 flex items-center justify-center p-4">
      <div className="w-full max-w-lg mx-auto">
        <button
          onClick={() => navigate('/')}
          className="mb-6 flex items-center gap-2 text-[#6B7280] hover:text-[#1F2937] transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6"/></svg>
          <span>{t('Back', 'पीछे')}</span>
        </button>

        <div className="flex justify-center gap-2 mb-8">
          {steps.map((s, i) => (
            <div
              key={s.id}
              className={`h-2 rounded-full transition-all duration-500 ${
                i === currentStep ? 'w-8 bg-[#0F766E]' : i < currentStep ? 'w-2 bg-[#0F766E]/40' : 'w-2 bg-gray-200'
              }`}
            />
          ))}
        </div>

        <h2 className="text-xl font-semibold text-[#1F2937] text-center mb-2">
          {t(flowName.en, flowName.hi)}
        </h2>

        <div className="bg-white rounded-3xl shadow-xl border border-gray-100 p-8 md:p-10">
          {renderStepContent()}

          {currentStepObj.id !== 'processing' && (
            <div className="mt-8 flex items-center justify-between">
              <div>
                {currentStepObj.skippable && (
                  <button
                    onClick={handleSkip}
                    className="text-sm text-[#6B7280] hover:text-[#1F2937] transition-colors"
                  >
                    {t('Skip this', 'इसे छोड़ें')}
                  </button>
                )}
              </div>
              <Button onClick={handleNext} size="lg">
                {currentStep === steps.length - 1 ? t('Analyze', 'विश्लेषण करें') : t('Next', 'आगे')}
              </Button>
            </div>
          )}
        </div>
      </div>

      <PanicButton />
    </div>
  );
}
