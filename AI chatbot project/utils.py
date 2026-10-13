"""
Utility functions for UI enhancements, animations, and formatting.
"""

import time
import sys
from datetime import datetime
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Color scheme for professional appearance
class Colors:
    PRIMARY = Fore.CYAN
    SECONDARY = Fore.MAGENTA
    SUCCESS = Fore.GREEN
    ERROR = Fore.RED
    WARNING = Fore.YELLOW
    INFO = Fore.BLUE
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT
    DIM = Style.DIM

def typing_animation(text, delay=0.03):
    """
    Simulate typing effect for realistic chatbot responses.
    
    Args:
        text (str): The message to display
        delay (float): Seconds between each character
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_banner():
    """Display an impressive ASCII art banner."""
    from pyfiglet import Figlet
    f = Figlet(font='slant')
    banner = f.renderText('NEXUS AI')
    
    print(Colors.PRIMARY + "=" * 60)
    print(Colors.SECONDARY + banner)
    print(Colors.PRIMARY + "=" * 60)
    print(Colors.INFO + "🤖 Your Intelligent Rule-Based Assistant")
    print(Colors.DIM + f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(Colors.PRIMARY + "=" * 60 + Colors.RESET)

def show_loading(message="Processing", duration=1):
    """Display a loading animation."""
    animation = "|/-\\"
    start_time = time.time()
    idx = 0
    
    while time.time() - start_time < duration:
        sys.stdout.write(f"\r{Colors.WARNING}{message} {animation[idx % len(animation)]}{Colors.RESET}")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write("\r" + " " * 50 + "\r")

def format_message(sender, message, is_bot=True):
    """
    Format messages with professional styling.
    
    Args:
        sender (str): Name of the sender
        message (str): Message content
        is_bot (bool): Whether the message is from the bot
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if is_bot:
        prefix = f"{Colors.SECONDARY}[{timestamp}] {Colors.BOLD}Nexus{Colors.RESET}"
        border = Colors.DIM + "└── " + Colors.RESET
    else:
        prefix = f"{Colors.INFO}[{timestamp}] {Colors.BOLD}You{Colors.RESET}"
        border = Colors.DIM + "└── " + Colors.RESET
    
    print(f"\n{prefix}")
    print(f"{border}{message}")
    print()

def clear_screen():
    """Clear the terminal screen for a clean interface."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    """Show available commands menu."""
    menu_text = f"""
{Colors.PRIMARY}╔══════════════════════════════════════════════════╗
║                    📋 COMMAND MENU                      ║
╠══════════════════════════════════════════════════════╣
║  {Colors.SUCCESS}💬 General Questions:{Colors.RESET}                         ║
║    • "How are you?"    • "What is your name?"          ║
║    • "What can you do?" • "Tell me a joke"             ║
║    • "What is AI?"      • "Help"                       ║
║                                                         ║
║  {Colors.WARNING}🎯 Special Features:{Colors.RESET}                         ║
║    • "Quote" - Motivational quote                      ║
║    • "Time" - Current date and time                    ║
║    • "Weather" - Weather small talk                    ║
║    • "Favorite color" - Personality test               ║
║                                                         ║
║  {Colors.ERROR}🚪 Exit Commands:{Colors.RESET}                            ║
║    • "bye", "exit", "quit", "goodbye"                  ║
║                                                         ║
║  {Colors.INFO}💡 Tip: Try asking about my personality!{Colors.RESET}          ║
╚══════════════════════════════════════════════════════╝{Colors.RESET}
    """
    print(menu_text)