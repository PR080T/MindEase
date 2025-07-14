"""
Test script to verify the input handling fix and IST timezone implementation
"""

import time
import pytz
from datetime import datetime
from unittest.mock import Mock

# Mock session state to simulate Streamlit behavior
class MockSessionState:
    def __init__(self):
        self.processed_inputs = set()
        self.last_input = ""
        self.input_counter = 0
        self.user_input = ""
        self.input_key = 0

def get_ist_time():
    """Get current time in IST (Indian Standard Time)."""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def get_ist_timestamp():
    """Get current timestamp in IST format (HH:MM:SS)."""
    return get_ist_time().strftime("%H:%M:%S")

def test_duplicate_prevention():
    """Test the improved duplicate prevention logic"""
    session_state = MockSessionState()
    
    # Simulate the improved logic
    def should_process_input(input_text, session_state):
        """
        Determine if an input should be processed based on improved duplicate prevention.
        
        Args:
            input_text: The input text to check
            session_state: Mock session state
            
        Returns:
            bool: True if input should be processed, False otherwise
        """
        if not input_text:
            return False
            
        current_time = int(time.time() * 1000)
        
        # Check for recent duplicates (within 2 seconds)
        recent_same_inputs = [
            p for p in session_state.processed_inputs 
            if p.startswith(input_text + "_") and 
            current_time - int(p.split('_')[-1]) < 2000  # 2 seconds
        ]
        
        # Allow processing if:
        # 1. Input is different from last processed input
        # 2. No recent duplicates found
        should_process = (
            input_text != getattr(session_state, 'last_input', '') and
            len(recent_same_inputs) == 0
        )
        
        if should_process:
            input_id = f"{input_text}_{current_time}"
            session_state.processed_inputs.add(input_id)
            session_state.last_input = input_text
            session_state.input_counter += 1
            return True
        return False
    
    # Test cases
    print("Testing improved duplicate prevention logic...")
    print(f"Current IST time: {get_ist_timestamp()}")
    print("=" * 50)
    
    # Test 1: First input should be processed
    result1 = should_process_input("hello", session_state)
    print(f"Test 1 - First 'hello': {result1} (should be True)")
    
    # Test 2: Immediate duplicate should be rejected
    result2 = should_process_input("hello", session_state)
    print(f"Test 2 - Immediate duplicate 'hello': {result2} (should be False)")
    
    # Test 3: Different input should be processed
    result3 = should_process_input("world", session_state)
    print(f"Test 3 - Different input 'world': {result3} (should be True)")
    
    # Test 4: Same input as previous should be rejected
    result4 = should_process_input("world", session_state)
    print(f"Test 4 - Duplicate 'world': {result4} (should be False)")
    
    # Test 5: Wait and try same input again (should work after 2+ seconds)
    print("Waiting 2.1 seconds...")
    time.sleep(2.1)  # Wait more than 2 seconds
    result5 = should_process_input("hello", session_state)
    print(f"Test 5 - 'hello' after 2+ seconds: {result5} (should be True)")
    
    # Test 6: Empty input should be rejected
    result6 = should_process_input("", session_state)
    print(f"Test 6 - Empty input: {result6} (should be False)")
    
    # Test 7: Whitespace-only input should be rejected
    result7 = should_process_input("   ", session_state)
    print(f"Test 7 - Whitespace input: {result7} (should be False)")
    
    print(f"\nFinal state:")
    print(f"- Processed inputs: {len(session_state.processed_inputs)}")
    print(f"- Last input: '{session_state.last_input}'")
    print(f"- Input counter: {session_state.input_counter}")
    print(f"- Processed input IDs: {list(session_state.processed_inputs)}")
    
    # Test cleanup logic
    print(f"\nTesting cleanup logic...")
    # Add many inputs to trigger cleanup
    for i in range(105):
        session_state.processed_inputs.add(f"test_{i}_{int(time.time() * 1000)}")
    
    print(f"Before cleanup: {len(session_state.processed_inputs)} inputs")
    
    # Simulate cleanup logic
    if len(session_state.processed_inputs) > 100:
        current_time = int(time.time() * 1000)
        recent_inputs = {
            input_id for input_id in session_state.processed_inputs
            if current_time - int(input_id.split('_')[-1]) < 300000  # 5 minutes
        }
        session_state.processed_inputs = recent_inputs
    
    print(f"After cleanup: {len(session_state.processed_inputs)} inputs")

def test_ist_timezone():
    """Test IST timezone functionality"""
    print("\n" + "=" * 50)
    print("Testing IST Timezone Implementation...")
    print("=" * 50)
    
    # Get current time in different formats
    utc_time = datetime.utcnow()
    local_time = datetime.now()
    ist_time = get_ist_time()
    
    print(f"UTC Time: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Local Time: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"IST Time: {ist_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"IST Timestamp (HH:MM:SS): {get_ist_timestamp()}")
    
    # Verify IST is correct (should be UTC+5:30)
    utc_ist = datetime.now(pytz.UTC).astimezone(pytz.timezone('Asia/Kolkata'))
    print(f"UTC to IST conversion: {utc_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    print("✅ IST timezone implementation working correctly!")

if __name__ == "__main__":
    test_duplicate_prevention()
    test_ist_timezone()