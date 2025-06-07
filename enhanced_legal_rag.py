#! python 
"""
Enhanced PDF Legal Q&A System - English Only
Handles complex legal questions with better reasoning
"""

import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import os
import re
from typing import List, Dict, Tuple

class EnhancedLegalRAG:
    def __init__(self):
        # Use a more powerful model for legal/technical content
        print("🔧 Initializing advanced legal RAG system...")
        
        # Try multiple embedding models in order of preference
        embedding_models = [
            "all-mpnet-base-v2",     # Best quality
            "all-MiniLM-L12-v2",     # Good balance
            "all-MiniLM-L6-v2"       # Fallback
        ]
        
        for model_name in embedding_models:
            try:
                print(f"🤖 Loading embedding model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                print(f"✅ Successfully loaded: {model_name}")
                break
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {str(e)}")
                continue
        else:
            raise Exception("Failed to load any embedding model")
        
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.keyword_index = {}
        
    def load_pdf(self) -> str:
        """Find and load PDF with better text extraction"""
        pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
        
        if not pdf_files:
            print("❌ No PDF files found!")
            return None
        
        # Use first PDF found
        pdf_file = pdf_files[0]
        print(f"📄 Loading: {pdf_file}")
        
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Clean and normalize text
                    page_text = self.clean_text(page_text)
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        return text
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning for better embedding quality"""
        # Remove page markers temporarily for processing
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Fix bullet points and numbered lists
        text = re.sub(r'[•·‣⁃]\s*', '• ', text)
        text = re.sub(r'(\d+)\.\s+', r'\1. ', text)
        text = re.sub(r'(\w+)\s*:\s*', r'\1: ', text)
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Normalize Arabic/English mixed text issues
        text = re.sub(r'([a-zA-Z])\s*([أ-ي])', r'\1 \2', text)
        text = re.sub(r'([أ-ي])\s*([a-zA-Z])', r'\1 \2', text)
        
        # Clean up legal references
        text = re.sub(r'(Article|Section|Chapter)\s+(\d+)', r'\1 \2', text, flags=re.IGNORECASE)
        return text.strip()
    
    def smart_chunk_text(self, text: str) -> List[Dict]:
        """Advanced chunking strategy with overlapping and semantic awareness"""
        print("✂️ Creating advanced semantic chunks...")
        
        chunks = []
        
        # Enhanced patterns for better legal document parsing
        legal_patterns = [
            r'(Article\s+\d+[.\s]*[:\-]?)',  # Article 1:
            r'(Section\s+\d+[.\s]*[:\-]?)',   # Section 1:
            r'(Chapter\s+\d+[.\s]*[:\-]?)',   # Chapter 1:
            r'(\d+\.\s*)',                    # 1. 
            r'(\(\d+\)\s*)',                  # (1) 
            r'([A-Z][^.!?]*[:]\s*)',         # TITLE:
        ]
        
        # Try hierarchical splitting
        main_sections = []
        current_text = text
        
        # First, split by major sections
        for pattern in legal_patterns:
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE | re.MULTILINE))
            if len(matches) >= 2:  # Found good structure
                sections = []
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(current_text)
                    section_text = current_text[start:end].strip()
                    if len(section_text.split()) > 20:  # Minimum size threshold
                        sections.append({
                            'text': section_text,
                            'header': match.group(1),
                            'type': 'legal_section'
                        })
                
                if sections:
                    main_sections = sections
                    break
        
        # If no good structure found, use semantic splitting
        if not main_sections:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            main_sections = [{'text': p, 'header': 'Paragraph', 'type': 'paragraph'} for p in paragraphs]
        
        # Process each section with overlapping chunks
        for section_idx, section in enumerate(main_sections):
            section_text = section['text']
            section_type = section['type']
            
            # Create base chunk
            base_chunk = {
                'text': section_text,
                'section_header': section['header'],
                'section_type': section_type,
                'section_index': section_idx,
                'page': self._extract_page_number(section_text),
                'word_count': len(section_text.split()),
            }
            
            # If section is large, split it further with overlap
            if len(section_text.split()) > 500:
                sentences = re.split(r'[.!?]+', section_text)
                current_chunk = ""
                sentence_start = 0
                
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                    
                    potential_chunk = current_chunk + sentence + ". "
                    
                    if len(potential_chunk.split()) > 400:
                        if current_chunk:
                            # Add main chunk
                            chunk_data = base_chunk.copy()
                            chunk_data.update({
                                'text': current_chunk,
                                'chunk_id': f"{section_idx}_{len(chunks)}",
                                'sentence_range': (sentence_start, i),
                                'semantic_type': self._classify_content(current_chunk)
                            })
                            chunks.append(chunk_data)
                            
                            # Create overlapping chunk (last 2 sentences + new content)
                            overlap_sentences = sentences[max(0, i-2):i+3]
                            overlap_text = ". ".join([s.strip() for s in overlap_sentences if s.strip()]) + "."
                            if len(overlap_text.split()) > 50:
                                overlap_chunk = base_chunk.copy()
                                overlap_chunk.update({
                                    'text': overlap_text,
                                    'chunk_id': f"{section_idx}_{len(chunks)}_overlap",
                                    'is_overlap': True,
                                    'semantic_type': self._classify_content(overlap_text)
                                })
                                chunks.append(overlap_chunk)
                        
                        current_chunk = sentence + ". "
                        sentence_start = i
                    else:
                        current_chunk = potential_chunk
                
                # Add remaining chunk
                if current_chunk and len(current_chunk.split()) > 30:
                    chunk_data = base_chunk.copy()
                    chunk_data.update({
                        'text': current_chunk,
                        'chunk_id': f"{section_idx}_{len(chunks)}",
                        'sentence_range': (sentence_start, len(sentences)),
                        'semantic_type': self._classify_content(current_chunk)
                    })
                    chunks.append(chunk_data)
            else:
                # Small section, add as single chunk
                base_chunk.update({
                    'chunk_id': f"{section_idx}_{len(chunks)}",
                    'semantic_type': self._classify_content(section_text)
                })
                chunks.append(base_chunk)
        
        # Add contextual chunks (combine related sections)
        contextual_chunks = self._create_contextual_chunks(main_sections)
        chunks.extend(contextual_chunks)
        print(f"📊 Created {len(chunks)} advanced chunks (including {len([c for c in chunks if c.get('is_overlap')])} overlapping chunks)")
        return chunks
    
    def create_chunk_metadata(self, text: str, page: int, section: str) -> Dict:
        """Create chunk with metadata"""
        return {
            'text': text,
            'page': page,
            'section': section,
            'word_count': len(text.split()),
            'has_articles': bool(re.search(r'Article\s+\d+', text, re.IGNORECASE)),
            'has_numbers': bool(re.search(r'\d+', text)),
        }
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from text"""
        page_match = re.search(r'--- Page (\d+) ---', text)
        return int(page_match.group(1)) if page_match else 1
    
    def _classify_content(self, text: str) -> str:
        """Classify content type for better retrieval"""
        text_lower = text.lower()
        
        if re.search(r'article\s+\d+', text_lower):
            return 'article'
        elif re.search(r'section\s+\d+', text_lower):
            return 'section'
        elif re.search(r'chapter\s+\d+', text_lower):
            return 'chapter'
        elif any(word in text_lower for word in ['shall', 'must', 'required', 'prohibited']):
            return 'obligation'
        elif any(word in text_lower for word in ['penalty', 'fine', 'punishment', 'violation']):
            return 'penalty'
        elif any(word in text_lower for word in ['procedure', 'process', 'steps', 'how to']):
            return 'procedure'
        elif any(word in text_lower for word in ['definition', 'means', 'refers to', 'includes']):
            return 'definition'
        else:
            return 'general'
    
    def _create_contextual_chunks(self, sections: List[Dict]) -> List[Dict]:
        """Create larger contextual chunks by combining related sections"""
        contextual_chunks = []
        
        # Group consecutive small sections
        i = 0
        while i < len(sections):
            if len(sections[i]['text'].split()) < 200:  # Small section
                combined_text = sections[i]['text']
                combined_headers = [sections[i]['header']]
                j = i + 1
                
                # Combine with next small sections
                while j < len(sections) and len(sections[j]['text'].split()) < 200 and len(combined_text.split()) < 600:
                    combined_text += "\n\n" + sections[j]['text']
                    combined_headers.append(sections[j]['header'])
                    j += 1
                
                if j > i + 1:  # We combined multiple sections
                    contextual_chunk = {
                        'text': combined_text,
                        'section_header': " + ".join(combined_headers),
                        'section_type': 'contextual',
                        'is_contextual': True,
                        'page': self._extract_page_number(combined_text),
                        'word_count': len(combined_text.split()),
                        'chunk_id': f"contextual_{len(contextual_chunks)}",
                        'semantic_type': 'combined'
                    }
                    contextual_chunks.append(contextual_chunk)
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        return contextual_chunks
    
    def process_document(self, text: str):
        """Process document - alias for setup_rag for API compatibility"""
        self.setup_rag(text)
    
    def setup_rag(self, text: str):
        """Setup enhanced RAG system with advanced embedding strategies"""
        print("🏗️ Building advanced vector store...")
        
        # Create smart chunks
        self.chunks = self.smart_chunk_text(text)
        
        # Prepare texts for embedding with enhanced preprocessing
        chunk_texts = []
        for chunk in self.chunks:
            # Create enhanced representation for embedding
            enhanced_text = self._create_enhanced_embedding_text(chunk)
            chunk_texts.append(enhanced_text)
        
        # Create embeddings in batches for better performance
        print("📊 Creating high-quality embeddings...")
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                show_progress_bar=True,
                normalize_embeddings=True,  # L2 normalization for better similarity
                convert_to_tensor=False
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Build enhanced vector store with better indexing
        print("🔍 Building optimized vector index...")
        dimension = embeddings.shape[1]
        
        # Use IndexHNSWFlat for better search quality (Hierarchical Navigable Small World)
        # This provides better recall than flat index
        self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is M parameter
        self.index.hnsw.efConstruction = 40  # Higher = better quality
        self.index.hnsw.efSearch = 32  # Higher = better search quality
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Create keyword index for hybrid search
        self._build_keyword_index()
        
        print(f"✅ Enhanced RAG system ready with {len(self.chunks)} chunks!")
        print(f"📈 Using HNSW index for superior search quality")
    
    def _create_enhanced_embedding_text(self, chunk: Dict) -> str:
        """Create enhanced text representation for better embeddings"""
        text = chunk['text']
        
        # Add semantic prefixes based on content type
        semantic_type = chunk.get('semantic_type', 'general')
        type_prefix = {
            'article': 'Legal Article: ',
            'section': 'Legal Section: ',
            'chapter': 'Legal Chapter: ',
            'obligation': 'Legal Obligation: ',
            'penalty': 'Legal Penalty: ',
            'procedure': 'Legal Procedure: ',
            'definition': 'Legal Definition: ',
            'general': 'Legal Content: '
        }.get(semantic_type, 'Legal Content: ')
        
        # Add section header if available
        header = chunk.get('section_header', '')
        if header and header != 'Paragraph':
            enhanced_text = f"{type_prefix}{header} - {text}"
        else:
            enhanced_text = f"{type_prefix}{text}"
        
        # Add contextual information
        if chunk.get('is_contextual'):
            enhanced_text = f"Combined Legal Sections: {enhanced_text}"
        
        return enhanced_text
    
    def _build_keyword_index(self):
        """Build keyword index for hybrid search"""
        print("🔤 Building keyword index for hybrid search...")
        
        # Extract keywords from each chunk
        self.keyword_index = {}
        for i, chunk in enumerate(self.chunks):
            text = chunk['text'].lower()
            
            # Extract important keywords
            keywords = set()
            
            # Legal terms
            legal_terms = re.findall(r'article\s+\d+|section\s+\d+|chapter\s+\d+', text)
            keywords.update(legal_terms)
            
            # Important words (nouns, verbs)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            keywords.update(words)
            
            # Store chunk indices for each keyword
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(i)
    def search_relevant_chunks(self, question: str, top_k: int = 10) -> List[Dict]:
        """Advanced hybrid search combining vector similarity and keyword matching"""
        if self.index is None:
            return []
        
        print(f"🔍 Performing hybrid search for: {question[:50]}...")
        
        # Vector similarity search
        vector_results = self._vector_search(question, top_k * 2)
        
        # Keyword search
        keyword_results = self._keyword_search(question, top_k)
        
        # Combine and rank results
        combined_results = self._combine_search_results(vector_results, keyword_results, question)
        
        # Return top results
        return combined_results[:top_k]
    
    def _vector_search(self, question: str, top_k: int) -> List[Dict]:
        """Perform vector similarity search"""
        # Create enhanced query
        enhanced_query = self._enhance_query_for_embedding(question)
        
        # Create query embedding
        query_embedding = self.embedding_model.encode(
            [enhanced_query], 
            normalize_embeddings=True
        )
        
        # Search with higher k for better recall
        search_k = min(top_k * 2, len(self.chunks))
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.1:  # Similarity threshold
                chunk = self.chunks[idx].copy()
                chunk['vector_score'] = float(score)
                chunk['search_type'] = 'vector'
                results.append(chunk)
        
        return results
    
    def _keyword_search(self, question: str, top_k: int) -> List[Dict]:
        """Perform keyword-based search"""
        if not hasattr(self, 'keyword_index'):
            return []
        
        question_lower = question.lower()
        keyword_matches = {}
        
        # Find exact keyword matches
        for keyword, chunk_indices in self.keyword_index.items():
            if keyword in question_lower:
                for idx in chunk_indices:
                    if idx not in keyword_matches:
                        keyword_matches[idx] = 0
                    keyword_matches[idx] += len(keyword)  # Weight by keyword length
        
        # Find partial matches for important terms
        important_words = re.findall(r'\b[a-zA-Z]{4,}\b', question_lower)
        for word in important_words:
            for keyword in self.keyword_index:
                if word in keyword or keyword in word:
                    for idx in self.keyword_index[keyword]:
                        if idx not in keyword_matches:
                            keyword_matches[idx] = 0
                        keyword_matches[idx] += 0.5  # Lower weight for partial matches
        
        # Convert to result format
        results = []
        for idx, score in sorted(keyword_matches.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['keyword_score'] = score
                chunk['search_type'] = 'keyword'
                results.append(chunk)
        
        return results
    
    def _enhance_query_for_embedding(self, question: str) -> str:
        """Enhance query for better embedding matching"""
        # Detect question type and add context
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['article', 'section', 'law']):
            prefix = "Legal article or section about: "
        elif any(word in question_lower for word in ['penalty', 'punishment', 'fine']):
            prefix = "Legal penalty or punishment for: "
        elif any(word in question_lower for word in ['procedure', 'process', 'how']):
            prefix = "Legal procedure or process for: "
        elif any(word in question_lower for word in ['definition', 'what is', 'meaning']):
            prefix = "Legal definition or meaning of: "
        else:
            prefix = "Legal information about: "
        
        return prefix + question
    
    def _combine_search_results(self, vector_results: List[Dict], keyword_results: List[Dict], question: str) -> List[Dict]:
        """Intelligently combine vector and keyword search results"""
        # Create unified result set
        all_results = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result.get('chunk_id', id(result))
            if chunk_id not in all_results:
                all_results[chunk_id] = result.copy()
            else:
                all_results[chunk_id].update(result)
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result.get('chunk_id', id(result))
            if chunk_id not in all_results:
                all_results[chunk_id] = result.copy()
            else:
                all_results[chunk_id].update(result)
        
        # Calculate combined scores
        for chunk_id, result in all_results.items():
            vector_score = result.get('vector_score', 0)
            keyword_score = result.get('keyword_score', 0)
            
            # Normalize keyword score
            if keyword_score > 0:
                keyword_score = min(keyword_score / 10.0, 1.0)
            
            # Combine scores with weights
            combined_score = (0.7 * vector_score) + (0.3 * keyword_score)
            
            # Boost score for exact content type matches
            question_lower = question.lower()
            semantic_type = result.get('semantic_type', '')
            
            if semantic_type == 'article' and 'article' in question_lower:
                combined_score *= 1.3
            elif semantic_type == 'penalty' and any(word in question_lower for word in ['penalty', 'punishment']):
                combined_score *= 1.3
            elif semantic_type == 'procedure' and any(word in question_lower for word in ['procedure', 'process', 'how']):
                combined_score *= 1.3
            
            result['combined_score'] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(all_results.values(), key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return sorted_results
    
    def enhanced_prompt_engineering(self, question: str, context: str) -> str:
        """Create enhanced prompt for complex legal questions"""
        
        # Detect question type
        is_case_scenario = any(word in question.lower() for word in ['if', 'case', 'scenario', 'situation', 'when', 'suppose'])
        is_article_request = any(word in question.lower() for word in ['article', 'section', 'law', 'rule', 'regulation'])
        is_implementation = any(word in question.lower() for word in ['implement', 'apply', 'enforce', 'should', 'must'])
        
        if is_case_scenario and is_article_request:
            prompt_type = "case_analysis"
        elif is_implementation:
            prompt_type = "implementation"
        else:
            prompt_type = "general"
        
        base_prompt = f"""
You are a legal expert analyzing a governance document. Answer the question based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
1. Use ONLY information from the context below
2. If the answer is not in the context, say "Information not available in the document"
3. Be precise and cite specific articles/sections when applicable
4. For case scenarios, identify the relevant articles and explain how they apply
5. Provide step-by-step reasoning for complex questions

Context from the document:
{context}

Question: {question}
"""

        if prompt_type == "case_analysis":
            prompt = base_prompt + """

For this case scenario question:
1. First identify the relevant articles/sections that apply
2. Explain what conditions or circumstances trigger each article
3. Determine which specific article(s) would be implemented
4. Provide the reasoning for your conclusion

Answer:"""

        elif prompt_type == "implementation":
            prompt = base_prompt + """

For this implementation question:
1. Identify the specific articles or rules that apply
2. Explain the conditions under which they are implemented
3. Describe the process or consequences
4. Cite specific sections from the document

Answer:"""

        else:
            prompt = base_prompt + """

Provide a comprehensive answer that:
1. Directly answers the question
2. Cites relevant articles or sections
3. Explains any important details or conditions

Answer:"""

        return prompt
    def ask_question(self, question: str) -> str:
        """Advanced question answering with intelligent context creation"""
        if self.index is None:
            return "System not initialized"
        
        print("🔍 Analyzing question and searching for optimal context...")
        
        # Get relevant chunks with advanced search
        relevant_chunks = self.search_relevant_chunks(question, top_k=12)
        
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        # Create intelligent context with chunk diversity
        context = self._create_intelligent_context(relevant_chunks, question)
        
        # Generate enhanced prompt
        prompt = self._create_advanced_prompt(question, context, relevant_chunks)
        
        print("🤖 Generating comprehensive answer...")
        
        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.1,  # Lower temperature for more focused responses
                    'top_p': 0.9,
                    'num_predict': 500,  # Allow longer responses
                }
            )
            
            answer = response['message']['content']
            
            # Enhance answer with source attribution
            enhanced_answer = self._enhance_answer_with_sources(answer, relevant_chunks[:5])
            
            return enhanced_answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _create_intelligent_context(self, chunks: List[Dict], question: str) -> str:
        """Create intelligent context prioritizing diverse and relevant information"""
        # Group chunks by type for diversity
        chunks_by_type = {}
        for chunk in chunks:
            chunk_type = chunk.get('semantic_type', 'general')
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append(chunk)
        
        # Select best chunks ensuring diversity
        selected_chunks = []
        max_chunks = 8
        
        # Prioritize different content types
        type_priority = ['article', 'section', 'obligation', 'penalty', 'procedure', 'definition', 'general']
        
        for chunk_type in type_priority:
            if chunk_type in chunks_by_type and len(selected_chunks) < max_chunks:
                # Take top chunks of this type
                type_chunks = chunks_by_type[chunk_type][:2]  # Max 2 per type
                selected_chunks.extend(type_chunks)
        
        # Fill remaining slots with highest-scoring chunks
        remaining_slots = max_chunks - len(selected_chunks)
        if remaining_slots > 0:
            remaining_chunks = [c for c in chunks if c not in selected_chunks][:remaining_slots]
            selected_chunks.extend(remaining_chunks)
        
        # Create context with rich formatting
        context_parts = []
        for i, chunk in enumerate(selected_chunks[:max_chunks]):
            header = chunk.get('section_header', f"Section {i+1}")
            page = chunk.get('page', 'Unknown')
            content = chunk['text']
            
            # Add semantic type indicator
            semantic_type = chunk.get('semantic_type', 'general').upper()
            
            # Format chunk with metadata
            chunk_context = f"[{semantic_type} - {header} - Page {page}]:\n{content}"
            context_parts.append(chunk_context)
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)
    
    def _create_advanced_prompt(self, question: str, context: str, chunks: List[Dict]) -> str:
        """Create advanced prompt with question-type awareness"""
        
        # Analyze question characteristics
        question_lower = question.lower()
        is_case_scenario = any(word in question_lower for word in ['if', 'case', 'scenario', 'situation', 'when', 'suppose'])
        is_article_lookup = any(word in question_lower for word in ['article', 'section', 'law', 'rule'])
        is_penalty_question = any(word in question_lower for word in ['penalty', 'punishment', 'fine', 'violation'])
        is_procedure_question = any(word in question_lower for word in ['how', 'procedure', 'process', 'steps'])
        is_definition_question = any(word in question_lower for word in ['what is', 'define', 'meaning', 'definition'])
        
        # Base instructions
        base_instructions = """You are an expert legal analyst specializing in governance documents. Your task is to provide accurate, comprehensive answers based STRICTLY on the provided document content.

CRITICAL REQUIREMENTS:
1. Use ONLY information from the provided context
2. If information is not available in the context, clearly state this
3. Cite specific articles, sections, or page numbers when referencing content
4. Provide direct quotes from the document when applicable
5. Be precise and avoid speculation or general legal knowledge"""
        
        # Question-specific instructions
        specific_instructions = ""
        
        if is_case_scenario and is_penalty_question:
            specific_instructions = """
For this scenario-based penalty question:
1. Identify the specific violation or action described
2. Find the exact article(s) that address this violation
3. Quote the relevant penalty or consequence from the document
4. Explain any conditions or circumstances that affect the penalty
5. Provide the complete legal reasoning chain"""
            
        elif is_case_scenario:
            specific_instructions = """
For this case scenario:
1. Break down the scenario into its key legal components
2. Identify which articles or sections apply to each component
3. Explain how the relevant articles should be interpreted in this context
4. Provide step-by-step legal reasoning
5. State the conclusion with supporting citations"""
            
        elif is_article_lookup:
            specific_instructions = """
For this article/section inquiry:
1. Locate the specific article(s) or section(s) mentioned
2. Provide the complete text of the relevant provisions
3. Explain the meaning and implications of these provisions
4. Mention any related articles that provide additional context"""
            
        elif is_penalty_question:
            specific_instructions = """
For this penalty/punishment question:
1. Identify the specific violation or offense
2. Find the exact penalty provisions in the document
3. Quote the penalty text directly from the document
4. Explain any factors that might affect the penalty severity
5. Mention any procedural requirements for imposing the penalty"""
            
        elif is_procedure_question:
            specific_instructions = """
For this procedural question:
1. Identify the specific process or procedure being asked about
2. Find the relevant procedural articles or sections
3. List the steps in chronological order as described in the document
4. Include any requirements, conditions, or timelines mentioned
5. Note any exceptions or special circumstances"""
            
        elif is_definition_question:
            specific_instructions = """
For this definition question:
1. Find the exact definition in the document
2. Quote the definition directly
3. Explain any clarifications or elaborations provided
4. Mention related terms or concepts if defined in the document"""
            
        else:
            specific_instructions = """
For this general inquiry:
1. Identify the main topic or subject of the question
2. Find all relevant provisions in the document
3. Organize the information logically
4. Provide comprehensive coverage of the topic
5. Include relevant context and related provisions"""
        
        # Source information
        source_info = f"""
Available sources: {len(chunks)} relevant sections found
Content types: {', '.join(set(c.get('semantic_type', 'general') for c in chunks[:5]))}
Page range: {min(c.get('page', 1) for c in chunks)} - {max(c.get('page', 1) for c in chunks)}"""
        
        # Construct final prompt
        prompt = f"""{base_instructions}

{specific_instructions}

{source_info}

DOCUMENT CONTENT:
{context}

QUESTION: {question}

ANALYSIS AND ANSWER:"""
        
        return prompt
    
    def _enhance_answer_with_sources(self, answer: str, top_chunks: List[Dict]) -> str:
        """Enhance answer with detailed source attribution"""
        
        # Extract unique sources
        sources = []
        seen_pages = set()
        
        for chunk in top_chunks:
            page = chunk.get('page', 'Unknown')
            section = chunk.get('section_header', 'Section')
            semantic_type = chunk.get('semantic_type', 'content')
            
            if page not in seen_pages:
                sources.append(f"Page {page} ({section} - {semantic_type.title()})")
                seen_pages.add(page)
        
        if sources:
            source_text = f"\n\n📚 **Sources Referenced:**\n• " + "\n• ".join(sources[:5])
            
            # Add search quality indicator
            avg_score = sum(c.get('combined_score', 0) for c in top_chunks) / len(top_chunks) if top_chunks else 0
            confidence_level = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
            source_text += f"\n\n🎯 **Search Confidence:** {confidence_level}"
            
            return answer + source_text
        
        return answer

def main():
    print("⚖️  Enhanced Legal PDF Q&A System")
    print("=" * 60)
    print("🎯 Optimized for complex legal questions")
    print("📚 Example: 'If I did X case then what article should be implemented on me?'")
    print("=" * 60)
    
    # Initialize system
    rag = EnhancedLegalRAG()
    
    # Load PDF
    text = rag.load_pdf()
    if not text:
        return
    
    # Setup RAG
    rag.setup_rag(text)
    
    print("\n💬 Ask complex legal questions (type 'quit' to exit):")
    print("💡 Try: 'If I violate governance rules, what article applies to me?'")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question:
                answer = rag.ask_question(question)
                print(f"\n📝 Answer:")
                print("-" * 50)
                print(answer)
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
