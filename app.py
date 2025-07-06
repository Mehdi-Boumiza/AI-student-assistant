import streamlit as st
import fitz  # PyMuPDF
import anthropic
import openai
from groq import Groq
import os
from typing import Dict, List, Optional
import json
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .question-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .summary-box {
        background: #f3e5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class AIStudyAssistant:
    """Main class for handling AI study assistant functionality"""
    
    def __init__(self):
        self.claude_client = None
        self.openai_client = None
        self.groq_client = None
        self.setup_ai_clients()
    
    def setup_ai_clients(self):
        """Initialize AI clients based on available API keys"""
        # Check for API keys in environment variables or Streamlit secrets
        try:
            if 'GROQ_API_KEY' in os.environ:
                self.groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
            elif hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                self.groq_client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        except Exception as e:
            st.sidebar.warning(f"Groq client setup failed: {e}")
        
        try:
            if 'ANTHROPIC_API_KEY' in os.environ:
                self.claude_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
            elif hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                self.claude_client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
        except Exception as e:
            st.sidebar.warning(f"Claude client setup failed: {e}")
        
        try:
            if 'OPENAI_API_KEY' in os.environ:
                self.openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            elif hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                self.openai_client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
        except Exception as e:
            st.sidebar.warning(f"OpenAI client setup failed: {e}")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file using PyMuPDF"""
        try:
            # Read the PDF file
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            
            pdf_document.close()
            return text.strip()
        
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def generate_study_content_groq(self, content: str) -> Dict:
        """Generate study content using Groq API"""
        try:
            prompt = f"""
            You are an AI study assistant helping high school students. Based on the following study material, please provide:

            1. A concise summary (2-3 paragraphs) highlighting the key concepts and main points
            2. 4-5 practice questions that test understanding of the material. Include a mix of:
               - Multiple choice questions (with 4 options each, mark the correct answer)
               - Short answer questions
               - One analytical/critical thinking question

            Study Material:
            {content}

            Please format your response as JSON with the following structure:
            {{
                "summary": "Your summary here",
                "questions": [
                    {{
                        "type": "multiple_choice",
                        "question": "Question text",
                        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                        "correct_answer": "A"
                    }},
                    {{
                        "type": "short_answer",
                        "question": "Question text",
                        "sample_answer": "Brief sample answer"
                    }}
                ]
            }}
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Fast and reliable model
                messages=[
                    {"role": "system", "content": "You are an expert AI study assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            # Try to parse JSON response
            try:
                response_content = response.choices[0].message.content
                # Extract JSON from response if it's wrapped in markdown
                if '```json' in response_content:
                    json_start = response_content.find('```json') + 7
                    json_end = response_content.find('```', json_start)
                    response_content = response_content[json_start:json_end]
                
                return json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback to simple parsing if JSON fails
                return self._parse_simple_response(response.choices[0].message.content)
        
        except Exception as e:
            st.error(f"Error generating content with Groq: {e}")
            return {"summary": "", "questions": []}
    
    def generate_study_content_claude(self, content: str) -> Dict:
        """Generate study content using Claude API"""
        try:
            prompt = f"""
            You are an AI study assistant helping high school students. Based on the following study material, please provide:

            1. A concise summary (2-3 paragraphs) highlighting the key concepts and main points
            2. 4-5 practice questions that test understanding of the material. Include a mix of:
               - Multiple choice questions (with 4 options each, mark the correct answer)
               - Short answer questions
               - One analytical/critical thinking question

            Study Material:
            {content}

            Please format your response as JSON with the following structure:
            {{
                "summary": "Your summary here",
                "questions": [
                    {{
                        "type": "multiple_choice",
                        "question": "Question text",
                        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                        "correct_answer": "A"
                    }},
                    {{
                        "type": "short_answer",
                        "question": "Question text",
                        "sample_answer": "Brief sample answer"
                    }}
                ]
            }}
            """
            
            message = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Try to parse JSON response
            try:
                response_content = message.content[0].text
                # Extract JSON from response if it's wrapped in markdown
                if '```json' in response_content:
                    json_start = response_content.find('```json') + 7
                    json_end = response_content.find('```', json_start)
                    response_content = response_content[json_start:json_end]
                
                return json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback to simple parsing if JSON fails
                return self._parse_simple_response(message.content[0].text)
        
        except Exception as e:
            st.error(f"Error generating content with Claude: {e}")
            return {"summary": "", "questions": []}
    
    def generate_study_content_openai(self, content: str) -> Dict:
        """Generate study content using OpenAI API"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI study assistant helping high school students."},
                    {"role": "user", "content": f"""
                    Based on the following study material, provide:
                    1. A concise summary (2-3 paragraphs)
                    2. 4-5 practice questions (mix of multiple choice and short answer)
                    
                    Study Material: {content}
                    
                    Format as JSON with 'summary' and 'questions' fields.
                    """}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return self._parse_simple_response(response.choices[0].message.content)
        
        except Exception as e:
            st.error(f"Error generating content with OpenAI: {e}")
            return {"summary": "", "questions": []}
    
    def _parse_simple_response(self, response: str) -> Dict:
        """Fallback parser for non-JSON responses"""
        # Simple fallback - split by common patterns
        parts = response.split('\n\n')
        summary = parts[0] if parts else "Summary not available"
        
        questions = []
        for i, part in enumerate(parts[1:], 1):
            if part.strip():
                questions.append({
                    "type": "short_answer",
                    "question": part.strip(),
                    "sample_answer": "Answer not provided"
                })
        
        return {"summary": summary, "questions": questions}
    
    def generate_study_content(self, content: str, ai_provider: str) -> Dict:
        """Generate study content using selected AI provider"""
        if ai_provider == "Groq" and self.groq_client:
            return self.generate_study_content_groq(content)
        elif ai_provider == "Claude" and self.claude_client:
            return self.generate_study_content_claude(content)
        elif ai_provider == "OpenAI" and self.openai_client:
            return self.generate_study_content_openai(content)
        else:
            st.error(f"Selected AI provider ({ai_provider}) is not available. Please check your API keys.")
            return {"summary": "", "questions": []}

def main():
    """Main Streamlit app function"""
    
    # Initialize the AI Study Assistant
    assistant = AIStudyAssistant()
    
    # App Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Study Assistant</h1>
        <p>Transform your study materials into summaries and practice questions!</p>
        <p><em>Built for Sonoma Hacks 4.0 🚀</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    st.sidebar.title("⚙️ Settings")
    
    # AI Provider Selection
    available_providers = []
    if assistant.groq_client:
        available_providers.append("Groq")
    if assistant.claude_client:
        available_providers.append("Claude")
    if assistant.openai_client:
        available_providers.append("OpenAI")
    
    if not available_providers:
        st.error("No AI providers available. Please set up your API keys!")
        st.info("""
        To use this app, you need to set up API keys:
        
        **Method 1: Environment Variables**
        - Set `GROQ_API_KEY` for Groq (Recommended - Super Fast!)
        - Set `ANTHROPIC_API_KEY` for Claude
        - Set `OPENAI_API_KEY` for OpenAI
        
        **Method 2: Streamlit Secrets**
        - Add keys to `.streamlit/secrets.toml`
        """)
        return
    
    ai_provider = st.sidebar.selectbox(
        "Choose AI Provider",
        available_providers,
        help="Select which AI service to use for generating study content"
    )
    
    # Input method selection
    st.sidebar.markdown("---")
    input_method = st.sidebar.radio(
        "📝 Input Method",
        ["Text Input", "Upload PDF"],
        help="Choose how you want to provide your study material"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📚 Study Material Input")
        
        content = ""
        
        if input_method == "Text Input":
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("**✏️ Paste your class notes or study material below:**")
            content = st.text_area(
                "Study Material",
                height=300,
                placeholder="Paste your notes, textbook excerpts, or any study material here...",
                help="The more detailed your input, the better the AI can help you study!"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif input_method == "Upload PDF":
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("**📄 Upload your PDF study material:**")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload lecture notes, textbook chapters, or study guides in PDF format"
            )
            
            if uploaded_file is not None:
                with st.spinner("📖 Extracting text from PDF..."):
                    content = assistant.extract_text_from_pdf(uploaded_file)
                
                if content:
                    st.success(f"✅ Successfully extracted {len(content)} characters from PDF")
                    with st.expander("📄 Preview extracted text"):
                        st.text_area("Extracted content", content[:1000] + "..." if len(content) > 1000 else content, height=200)
                else:
                    st.error("❌ Could not extract text from PDF. Please try a different file.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 How it works")
        st.markdown("""
        1. **📝 Input**: Paste text or upload PDF
        2. **🤖 AI Processing**: Generate summary & questions
        3. **📊 Study**: Review and practice!
        """)
        
        st.markdown("### ✨ Features")
        st.markdown("""
        - 📋 **Smart Summaries**
        - ❓ **Practice Questions**
        - ⚡ **Lightning Fast with Groq**
        - 📱 **Mobile Friendly**
        - 🚀 **Perfect for Hackathons**
        """)
    
    # Generate study content
    if content and len(content.strip()) > 50:
        st.markdown("---")
        
        if st.button("🚀 Generate Study Help", type="primary", use_container_width=True):
            with st.spinner(f"🧠 Generating study content using {ai_provider}..."):
                result = assistant.generate_study_content(content, ai_provider)
            
            if result["summary"] or result["questions"]:
                st.success("✅ Study content generated successfully!")
                
                # Display Summary
                if result["summary"]:
                    st.markdown("### 📋 Summary")
                    st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)
                
                # Display Questions
                if result["questions"]:
                    st.markdown("### ❓ Practice Questions")
                    
                    for i, question in enumerate(result["questions"], 1):
                        st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                        st.markdown(f"**Question {i}:**")
                        st.markdown(question["question"])
                        
                        if question["type"] == "multiple_choice" and "options" in question:
                            for option in question["options"]:
                                st.markdown(f"- {option}")
                            if "correct_answer" in question:
                                st.markdown(f"*Correct answer: {question['correct_answer']}*")
                        
                        elif question["type"] == "short_answer" and "sample_answer" in question:
                            with st.expander("💡 Sample Answer"):
                                st.markdown(question["sample_answer"])
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Export option (bonus feature)
                st.markdown("---")
                st.markdown("### 💾 Export Options")
                
                # Create downloadable content
                export_content = f"""
# Study Summary

{result['summary']}

# Practice Questions

"""
                for i, question in enumerate(result["questions"], 1):
                    export_content += f"\n## Question {i}\n{question['question']}\n"
                    if question["type"] == "multiple_choice" and "options" in question:
                        for option in question["options"]:
                            export_content += f"{option}\n"
                        if "correct_answer" in question:
                            export_content += f"\n**Correct Answer:** {question['correct_answer']}\n"
                    elif "sample_answer" in question:
                        export_content += f"\n**Sample Answer:** {question['sample_answer']}\n"
                
                st.download_button(
                    label="📥 Download Study Guide",
                    data=export_content,
                    file_name="study_guide.txt",
                    mime="text/plain",
                    help="Download your summary and questions as a text file"
                )
    
    elif content and len(content.strip()) <= 50:
        st.warning("⚠️ Please provide more content (at least 50 characters) for better results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎓 Built for <strong>Sonoma Hacks 4.0</strong> | Empowering students with AI</p>
        <p>Made with ❤️ using Streamlit & Groq ⚡</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
