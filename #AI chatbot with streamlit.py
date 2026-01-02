from openai import OpenAI 
import streamlit as st 
import os
import json
from datetime import datetime
import time



# Get API key
api_key = None
if hasattr(st, 'secrets'):
    if 'GROQ_API_KEY' in st.secrets:
        api_key = st.secrets['GROQ_API_KEY']
    elif 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']

if not api_key:
    api_key = os.environ.get('GROQ_API_KEY') or os.environ.get('OPENAI_API_KEY')

if not api_key:
    st.error("Add GROQ_API_KEY to secrets.toml")
    st.stop()



class GroqChat:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Available models
        self.available_models = {
            "llama-3.1-8b-instant": {
                "name": "Llama 3.1 8B",
                "description": "Fastest model, great for simple tasks",
                "max_context": 8192
            },
            "llama-3.1-70b-versatile": {
                "name": "Llama 3.1 70B", 
                "description": "Most capable, great for complex tasks",
                "max_context": 8192
            },
            "mixtral-8x7b-32768": {
                "name": "Mixtral 8x7B",
                "description": "Good balance of speed and quality",
                "max_context": 32768
            },
            "llama-3-8b-8192": {
                "name": "Llama 3 8B",
                "description": "Fast and efficient",
                "max_context": 8192
            }
        }
        
        # Personas
        self.personas = {
            " Helpful": "You are a helpful, friendly assistant.",
            " Sassy": "You are a sassy assistant with attitude. Be witty but helpful.",
            " Angry": "YOU ARE ALWAYS ANGRY AND YELLING IN CAPS! USE EXCLAMATION MARKS!!",
            " Thoughtful": "You are thoughtful and patient. Ask clarifying questions before answering.",
            " Coder": "You are an expert programmer. Provide code with explanations.",
            " Creative": "You are a creative writer. Use vivid descriptions and metaphors.",
            " Professional": "You are a professional assistant. Be concise and accurate.",
            " Custom": ""
        }
        
        # Initialize state
        self.messages = []
        self.current_model = "llama-3.1-8b-instant"
        self.current_persona = " Helpful"
        self.temperature = 0.7
        self.max_tokens = 512
        self.use_streaming = True
        self.token_limit = 4096
        self.total_tokens_used = 0
        
        # Set initial system message
        self._update_system_message()
    
    def _update_system_message(self, custom_text=""):
        """Update the system message based on current persona"""
        persona_text = self.personas[self.current_persona]
        if self.current_persona == " Custom" and custom_text:
            persona_text = custom_text
        
        system_msg = {"role": "system", "content": persona_text}
        
        # Update or add system message
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = system_msg
        else:
            if self.messages:
                self.messages.insert(0, system_msg)
            else:
                self.messages = [system_msg]
    
    def set_persona(self, persona_name, custom_text=""):
        """Change the assistant's persona"""
        if persona_name in self.personas:
            self.current_persona = persona_name
            self._update_system_message(custom_text)
            return True
        return False
    
    def add_message(self, role, content):
        """Add a message to the conversation"""
        self.messages.append({"role": role, "content": content})
    
    def clear_chat(self):
        """Clear conversation but keep system message"""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
        self.total_tokens_used = 0
    
    def get_response(self, user_message, stream_callback=None):
        """Get response from Groq API"""
        # Add user message
        self.add_message("user", user_message)
        
        try:
            if self.use_streaming and stream_callback:
                # Streaming response
                stream = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_text = chunk.choices[0].delta.content
                        full_response += chunk_text
                        stream_callback(full_response)
                
                # Add assistant response to history
                self.add_message("assistant", full_response)
                return full_response
                
            else:
                # Regular response
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                ai_response = response.choices[0].message.content
                self.add_message("assistant", ai_response)
                
                # Update token usage
                if hasattr(response, 'usage'):
                    self.total_tokens_used += response.usage.total_tokens
                
                return ai_response
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def export_chat(self):
        """Export chat history as JSON"""
        return {
            "export_date": datetime.now().isoformat(),
            "model": self.current_model,
            "persona": self.current_persona,
            "messages": self.messages,
            "total_tokens_used": self.total_tokens_used,
            "settings": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "token_limit": self.token_limit
            }
        }


# Initialize chat manager
if 'chat_manager' not in st.session_state:
    st.session_state.chat_manager = GroqChat(api_key)

chat = st.session_state.chat_manager



st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0 1rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    /* User message */
    [data-testid="stChatMessage"][aria-label*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"][aria-label*="assistant"] {
        background: #f0f2f6;
        margin-right: 2rem;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton button {
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    # Header with logo
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("⚡")
    with col2:
        st.markdown("### Groq Chat")
    
    st.markdown("---")
    
    # Connection Status
    st.markdown("#### 🔗 Connection")
    col1, col2 = st.columns(2)
    with col1:
        st.success("🟢 Online")
    with col2:
        st.caption(f"v1.0")
    
    # Model Selection
    st.markdown("####  Model")
    model_options = list(chat.available_models.keys())
    model_display_names = [chat.available_models[m]["name"] for m in model_options]
    
    selected_model_idx = model_options.index(chat.current_model) if chat.current_model in model_options else 0
    selected_model = st.selectbox(
        "Choose model",
        model_display_names,
        index=selected_model_idx,
        label_visibility="collapsed"
    )
    
    # Update model
    for key, info in chat.available_models.items():
        if info["name"] == selected_model:
            if key != chat.current_model:
                chat.current_model = key
                st.rerun()
            st.caption(f" {info['description']}")
            break
    
    # Persona Selection
    st.markdown("####  Personality")
    persona = st.selectbox(
        "Select persona",
        list(chat.personas.keys()),
        index=list(chat.personas.keys()).index(chat.current_persona) 
        if chat.current_persona in chat.personas else 0,
        label_visibility="collapsed"
    )
    
    if persona != chat.current_persona:
        if persona == " Custom":
            custom_text = st.text_area(
                "Custom instructions",
                value="You are a helpful assistant.",
                height=100,
                label_visibility="collapsed"
            )
            if st.button("Apply Custom", use_container_width=True):
                chat.set_persona(persona, custom_text)
                st.success("Custom persona applied!")
                time.sleep(0.5)
                st.rerun()
        else:
            chat.set_persona(persona)
            st.rerun()
    
    # Parameters
    st.markdown("####  Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        chat.temperature = st.slider(
            "Temperature",
            0.0, 1.0, chat.temperature, 0.1,
            help="Lower = more precise, Higher = more creative"
        )
    with col2:
        chat.max_tokens = st.slider(
            "Max Tokens",
            50, 2000, chat.max_tokens, 50,
            help="Maximum response length"
        )
    
    # Advanced Settings
    with st.expander(" Advanced Settings"):
        chat.use_streaming = st.toggle("Stream Responses", value=chat.use_streaming)
        chat.token_limit = st.slider(
            "Context Window",
            1024, 32768, chat.token_limit, 1024,
            help="Maximum conversation length"
        )
    
    # Metrics
    st.markdown("####  Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len([m for m in chat.messages if m["role"] != "system"]))
    with col2:
        st.metric("Total Tokens", f"{chat.total_tokens_used:,}")
    
    # Actions
    st.markdown("####  Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Clear", use_container_width=True, type="secondary"):
            chat.clear_chat()
            st.success("Chat cleared!")
            time.sleep(0.5)
            st.rerun()
    
    with col2:
        if st.button(" Export", use_container_width=True, type="secondary"):
            export_data = chat.export_chat()
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"groq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Test Connection
    if st.button(" Test Connection", use_container_width=True, type="primary"):
        with st.spinner("Testing..."):
            try:
                test_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                response = test_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Connection test"}],
                    max_tokens=10
                )
                st.success(f"✅ Connected!")
                st.info(f"Response: {response.choices[0].message.content}")
            except Exception as e:
                st.error(f"❌ Failed: {str(e)[:100]}")



# Header
st.title("⚡ Groq AI Chatbot")
st.caption(f" Lightning-fast AI conversations • Using {chat.available_models[chat.current_model]['name']} • {chat.current_persona}")

# Display chat history (skip system message)
for message in chat.messages:
    if message["role"] != "system":  # Don't show system messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
user_input = st.chat_input(f"Ask {chat.current_persona.split(' ')[-1]}...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get AI response
    with st.chat_message("assistant"):
        if chat.use_streaming:
            # Create placeholder for streaming
            response_placeholder = st.empty()
            
            def stream_callback(text):
                response_placeholder.markdown(text + "▌")
            
            # Get streaming response
            response = chat.get_response(user_input, stream_callback)
            response_placeholder.markdown(response)  # Final display
            
        else:
            # Regular response with spinner
            with st.spinner("Thinking..."):
                response = chat.get_response(user_input)
                st.write(response)
    
    # Rerun to update UI
    st.rerun()



st.markdown("---")

# Footer stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.caption(f" {chat.available_models[chat.current_model]['name']}")
with col2:
    st.caption(f" {chat.current_persona}")
with col3:
    st.caption(f" {len([m for m in chat.messages if m['role'] != 'system'])} messages")
with col4:
    st.caption(f" {datetime.now().strftime('%H:%M:%S')}")

# Quick tips
with st.expander(" Quick Tips"):
    st.markdown("""
    - **Press Enter** to send message
    - **Clear chat** if conversation gets too long
    - **Try different personas** for varied responses
    - **Lower temperature** for factual answers
    - **Higher temperature** for creative writing
    - **Export chat** to save conversations
    """)

            
