import groq
from groq import Groq

class Translator:
    def __init__(self, api_key):
        if not api_key:
            self.client = None
            print("⚠️ Groq API key not provided. Translation will be unavailable.")
        else:
            try:
                self.client = Groq(api_key=api_key)
                print("✅ Groq translator initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Groq: {e}")
                self.client = None
    
    def translate_text(self, text, source_lang, target_lang):
        """Translate text using Groq API with updated model"""
        if not self.client:
            return "Translation service not available. Please configure GROQ_API_KEY."
        
        if not text or not text.strip():
            return ""
        
        try:
            # Language mapping
            lang_names = {
                'en': 'English',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'hi': 'Hindi',
                'mr': 'Marathi',
                'zh': 'Chinese (Simplified)',
                'ja': 'Japanese',
                'ko': 'Korean',
                'ar': 'Arabic',
                'ru': 'Russian',
                'auto': 'Auto-detect'
            }
            
            source_name = lang_names.get(source_lang, source_lang)
            target_name = lang_names.get(target_lang, target_lang)
            
            # Create a more detailed prompt for better translation
            if source_lang == 'auto':
                prompt = f"""You are a professional translator. 
First, detect the language of the following text, then translate it to {target_name}.
Maintain the original meaning, tone, and formatting.

Text to translate:
{text}

Please provide ONLY the translated text in {target_name}, nothing else."""
            else:
                prompt = f"""You are a professional translator.
Translate the following text from {source_name} to {target_name}.
Maintain the original meaning, tone, and formatting.

Text to translate:
{text}

Please provide ONLY the translated text in {target_name}, nothing else."""
            
            # Make API call with updated model
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate text accurately to {target_name}."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # Updated to current available model
                temperature=0.3,  # Lower temperature for more accurate translation
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            
            translated = chat_completion.choices[0].message.content.strip()
            print(f"✅ Translation successful: {source_name} → {target_name}")
            return translated
            
        except groq.APIError as e:
            print(f"❌ Groq API error: {str(e)}")
            return f"Translation failed due to API error: {str(e)}\n\nOriginal text:\n{text}"
        except Exception as e:
            print(f"❌ Translation error: {str(e)}")
            return f"Translation failed: {str(e)}\n\nOriginal text:\n{text}"