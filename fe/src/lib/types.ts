export type Language = 'en' | 'hi';
export type RiskLevel = 'low' | 'medium' | 'high';
export type UrgencyLevel = 'low' | 'medium' | 'high';
export type DocumentType = 'notice' | 'court-order' | 'agreement' | 'other';
export type FrameworkPartKey = 'what' | 'happened' | 'seriousness' | 'action' | 'avoid-cheated' | 'prevent' | 'legal-basis' | 'conclusion';

export interface Helpline {
  id: string;
  name: string;
  nameHi: string;
  number: string;
  description: string;
  descriptionHi: string;
}

export interface TileData {
  id: string;
  emoji: string;
  titleEn: string;
  titleHi: string;
  bgClass: string;
  route: string;
}

export interface FrameworkPart {
  key: FrameworkPartKey;
  emoji: string;
  titleEn: string;
  titleHi: string;
  contentEn: string;
  contentHi: string;
  riskLevel?: RiskLevel;
}

export interface CaseSummary {
  id: string;
  title: string;
  date: string;
  riskLevel: RiskLevel;
  documents: DocumentInfo[];
  framework: FrameworkPart[];
  messages: Message[];
}

export interface DocumentInfo {
  id: string;
  name: string;
  type: DocumentType;
  uploadDate: string;
  filePath?: string;
  size?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  suggestions?: string[];
}

export interface AppSettings {
  language: Language;
  fontSize: number;
  highContrast: boolean;
  voiceInput: boolean;
  readAloudVoice: string;
  dataLocation: string;
}

export interface AntiExploitationTool {
  id: string;
  emoji: string;
  titleEn: string;
  titleHi: string;
  descriptionEn: string;
  descriptionHi: string;
}
