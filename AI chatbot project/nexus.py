#!/usr/bin/env python3
"""
NEXUS - Professional Rule-Based AI Chatbot
Artificial Intelligence Internship Project

A sophisticated rule-based chatbot using structured if-else logic,
pattern matching, and an engaging personality.
"""

import sys
from utils import *
from responses import ResponseManager

class NexusChatbot:
    """Main chatbot class managing conversation flow and UI."""
    
    def __init__(self):
        """Initialize the chatbot with response manager."""
        self.response_manager = ResponseManager()
        self.running = True
        self.conversation_count = 0
        
    def process_user_input(self, user_input):
        """
        Process user input and generate appropriate response.
        
        Args:
            user_input (str): Raw user input
            
        Returns:
            str: Bot response
        """
        # Handle empty input
        if not user_input or user_input.isspace():
            return "I didn't catch that. Could you say something? 🎤"
        
        # Check for exit commands
        if user_input.lower() in ['bye', 'exit', 'quit', 'goodbye']:
            self.running = False
            return None
        
        # Get response from rule-based engine
        response = self.response_manager.get_response(user_input)
        
        # Increment conversation counter
        self.conversation_count += 1
        
        return response
    
    def run(self):
        """Main conversation loop with enhanced UI."""
        # Clear screen and show banner
        clear_screen()
        print_banner()
        
        # Welcome message
        welcome_msg = "✨ Welcome to Nexus! I'm your intelligent rule-based assistant. ✨\nType 'Help' anytime to see my capabilities, or 'bye' to exit."
        typing_animation(welcome_msg, 0.02)
        print()
        
        # Main conversation loop
        while self.running:
            try:
                # Get user input with professional prompt
                user_input = input(f"{Colors.INFO}{Colors.BOLD}You ›{Colors.RESET} ")
                
                # Process input with loading animation
                show_loading("Processing", 0.5)
                response = self.process_user_input(user_input)
                
                # Check if we should exit
                if response is None:
                    farewell = random.choice([
                        "Thanks for chatting! Have an amazing day! 👋",
                        "Goodbye! Come back soon! 🤖",
                        "Farewell, human friend! Until next time! 🌟"
                    ])
                    typing_animation(f"\n{Colors.SECONDARY}Nexus ›{Colors.RESET} {farewell}", 0.03)
                    print(f"\n{Colors.DIM}{'='*60}{Colors.RESET}")
                    print(f"{Colors.SUCCESS}📊 Session Stats: {self.conversation_count} exchanges{Colors.RESET}")
                    print(f"{Colors.INFO}💡 Tip: Run again to start a new conversation!{Colors.RESET}")
                    print(f"{Colors.DIM}{'='*60}{Colors.RESET}\n")
                    break
                
                # Display formatted response
                typing_animation(f"\n{Colors.SECONDARY}Nexus ›{Colors.RESET} {response}", 0.025)
                print()
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}Goodbye! Thanks for using Nexus! 👋{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.ERROR}⚠️ An error occurred: {e}{Colors.RESET}")
                print("Please try again! 😊")

def main():
    """Entry point for the chatbot application."""
    # Check Python version
    if sys.version_info < (3, 6):
        print("⚠️ Please use Python 3.6 or higher for the best experience.")
        sys.exit(1)
    
    # Run chatbot
    chatbot = NexusChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()