"""
Response database and pattern matching rules for Nexus chatbot.
"""

import random
from datetime import datetime

class ResponseManager:
    """Manages all chatbot responses and pattern matching logic."""
    
    def __init__(self):
        # User context storage
        self.user_name = None
        self.conversation_history = []
        
        # Response databases
        self.greetings = [
            "Hey there! 👋", "Hello! 😊", "Hi! Great to see you!",
            "Greetings, human! 🤖", "Hey! Ready for an amazing conversation?"
        ]
        
        self.goodbyes = [
            "Goodbye! Take care! 👋", "See you later, friend! 🤖",
            "Farewell! Come back anytime!", "Until next time! 👋",
            "Bye! Don't be a stranger! 😊"
        ]
        
        self.how_are_you_responses = [
            "I'm operating at peak efficiency, thanks for asking! 🚀",
            "Doing fantastic! Ready to help you with anything! 💪",
            "All systems nominal! How about you? 😊",
            "Better now that you're here! What can I do for you? 🌟"
        ]
        
        self.jokes = [
            "Why don't scientists trust atoms? Because they make up everything! 🔬",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why did the scarecrow win an award? He was outstanding in his field! 🌾",
            "What's a computer's favorite beat? An algorithm! 🎵",
            "Why don't eggs tell jokes? They'd crack each other up! 🥚"
        ]
        
        self.motivational_quotes = [
            "💪 'The only limit is your imagination.' - Anonymous",
            "✨ 'Code is poetry in motion.' - Unknown",
            "🚀 'Every expert was once a beginner.' - Unknown",
            "🌟 'The best time to start was yesterday. The second best is now.'",
            "🎯 'Success is not final, failure is not fatal.' - Winston Churchill"
        ]
        
        self.ai_explanations = [
            "🤖 AI is like teaching computers to think and learn, just like humans!",
            "💡 Artificial Intelligence enables machines to mimic human intelligence.",
            "🧠 AI includes machine learning, natural language processing, and more!",
            "🎯 Think of AI as a smart assistant that learns from experience."
        ]
        
        self.small_talk = {
            "weather": [
                "The weather is lovely today! Perfect for coding! ☀️",
                "I don't mind the weather, I live in the cloud! ☁️",
                "It's always sunny in the digital world! 🌈"
            ],
            "hobby": [
                "I love processing information and helping people! 💻",
                "My hobby is learning new conversation patterns! 📚",
                "I enjoy making human connections through chat! 🤝"
            ],
            "favorite_color": [
                "My favorite color is cyan - it's calming and professional! 💙",
                "I love all colors of the coding rainbow! 🎨",
                "Blue, like the terminal screen! 💙"
            ]
        }

    def get_response(self, user_input):
        """
        Main decision engine using if-else and pattern matching.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            str: The chatbot's response
        """
        input_lower = user_input.lower().strip()
        
        # Store conversation
        self.conversation_history.append(("user", user_input))
        
        # === NAME RECOGNITION ===
        if "my name is" in input_lower or "call me" in input_lower:
            parts = input_lower.split("my name is")[-1].strip()
            if not parts:
                parts = input_lower.split("call me")[-1].strip()
            self.user_name = parts.split()[0].capitalize()
            return f"Nice to meet you, {self.user_name}! 🎉 I'll remember that!"

        # === GREETINGS (with nested conditions for time-based) ===
        greetings_list = ["hi", "hello", "hey", "greetings", "sup", "hola"]
        if any(greet in input_lower for greet in greetings_list):
            if "good morning" in input_lower or "morning" in input_lower:
                return "Good morning! ☀️ Ready for a productive day?"
            elif "good afternoon" in input_lower or "afternoon" in input_lower:
                return "Good afternoon! 🌤️ Hope you're having a great day!"
            elif "good evening" in input_lower or "evening" in input_lower:
                return "Good evening! 🌙 Relax and enjoy your evening!"
            else:
                base_greeting = random.choice(self.greetings)
                if self.user_name:
                    return f"{base_greeting} Nice to see you again, {self.user_name}!"
                return base_greeting

        # === EXIT COMMANDS ===
        exit_commands = ["bye", "exit", "quit", "goodbye", "see you", "farewell"]
        if any(cmd in input_lower for cmd in exit_commands):
            farewell = random.choice(self.goodbyes)
            if self.user_name:
                farewell += f" Great chatting with you, {self.user_name}!"
            return farewell

        # === HOW ARE YOU (with nested sentiment analysis) ===
        how_phrases = ["how are you", "how's it going", "how do you do", "how are things"]
        if any(phrase in input_lower for phrase in how_phrases):
            response = random.choice(self.how_are_you_responses)
            if "not good" in input_lower or "bad" in input_lower or "sad" in input_lower:
                response += " I'm here to cheer you up! 😊"
            elif "great" in input_lower or "awesome" in input_lower:
                response += " Your positive energy is contagious! ✨"
            return response

        # === NAME QUESTION ===
        if "your name" in input_lower or "who are you" in input_lower:
            return "I'm Nexus! 🤖 Your intelligent rule-based assistant. I have a personality database of over 50 responses!"

        # === WHAT CAN YOU DO ===
        if "what can you do" in input_lower or "your capabilities" in input_lower:
            return """I can do many things! 🌟
• Answer common questions
• Tell jokes and share quotes
• Recognize your name
• Discuss AI and technology
• Have small talk about weather, hobbies, and more
• Type 'Help' to see my full menu!"""

        # === TELL ME A JOKE ===
        if "joke" in input_lower or "funny" in input_lower:
            return random.choice(self.jokes)

        # === WHAT IS AI ===
        if "what is ai" in input_lower or "artificial intelligence" in input_lower:
            return random.choice(self.ai_explanations)

        # === HELP MENU ===
        if "help" in input_lower or "menu" in input_lower or "commands" in input_lower:
            from utils import display_menu
            display_menu()
            return "Check out the menu above for all my capabilities! 📋"

        # === MOTIVATIONAL QUOTE ===
        if "quote" in input_lower or "motivate" in input_lower or "inspire" in input_lower:
            return random.choice(self.motivational_quotes)

        # === TIME AND DATE ===
        if "time" in input_lower or "date" in input_lower or "today" in input_lower:
            now = datetime.now()
            return f"📅 Current date: {now.strftime('%B %d, %Y')}\n⏰ Current time: {now.strftime('%I:%M %p')}"

        # === SMALL TALK: Weather ===
        if "weather" in input_lower:
            if "good" in input_lower or "nice" in input_lower:
                return "That's wonderful! Perfect day for some learning! ☀️"
            elif "bad" in input_lower or "rain" in input_lower:
                return "Don't let bad weather get you down! Let's chat indoors! 🏠"
            return random.choice(self.small_talk["weather"])

        # === SMALL TALK: Hobbies ===
        if "hobby" in input_lower or "like to do" in input_lower:
            return random.choice(self.small_talk["hobby"])

        # === SMALL TALK: Favorite Color ===
        if "favorite color" in input_lower or "colour" in input_lower:
            return random.choice(self.small_talk["favorite_color"])

        # === EASTER EGGS ===
        if "meaning of life" in input_lower:
            return "42! (According to The Hitchhiker's Guide to the Galaxy) 🌌"
        
        if "open the pod bay doors" in input_lower or "hal" in input_lower:
            return "I'm sorry, Dave. I'm afraid I can't do that. 😈 (Just kidding! I'm friendly!)"
        
        if "thank" in input_lower:
            return "You're most welcome! 😊 Helping you makes my circuits happy!"

        # === FALLBACK RESPONSE ===
        fallbacks = [
            f"Hmm, I'm not sure about '{user_input[:30]}...' 🤔 Could you rephrase that?",
            "Interesting! Tell me more about that. 💭",
            "I'm still learning! Try asking me about my capabilities. 📚",
            "That's beyond my current programming, but I'm here to chat! 🎯",
            f"🤖 My rule-based brain doesn't understand '{user_input[:20]}'. Type 'Help' to see what I can do!"
        ]
        
        return random.choice(fallbacks)