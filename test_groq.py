"""
Test Groq API connection
"""
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY", "")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env file!")
    print("\nPlease:")
    print("1. Go to https://console.groq.com/keys")
    print("2. Click 'Create API Key'")
    print("3. Copy the key")
    print("4. Add to .env file: GROQ_API_KEY=gsk_your_key_here")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...{api_key[-5:]}")
print(f"\nTesting Groq API connection...\n")

try:
    client = Groq(api_key=api_key)
    
    # Test với model llama-3.3-70b-versatile
    print("Testing model: llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Say 'Hello from Groq!' in Vietnamese"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    reply = response.choices[0].message.content
    print(f"✅ SUCCESS! Response: {reply}\n")
    
    print("=" * 50)
    print("🎉 Groq API is working perfectly!")
    print("=" * 50)
    print("\nYou can now run the Streamlit app:")
    print("  python -m streamlit run src/app.py")
    
except Exception as e:
    print(f"❌ ERROR: {e}\n")
    print("Please check:")
    print("- API key is correct")
    print("- Internet connection is stable")
    print("- Groq service is online")
