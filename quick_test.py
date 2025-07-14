#!/usr/bin/env python3

# Quick test to verify the new functionality works
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from MentalHealthChatbot import detect_specific_issue, get_specific_issue_response
    
    # Test the original sleep issue
    print("Testing original issue: 'i cant fall asleep'")
    issue = detect_specific_issue('i cant fall asleep')
    print(f"Detected issue: {issue}")
    
    if issue != 'none':
        response = get_specific_issue_response(issue)
        print(f"Response: {response[:150]}...")
    
    print("\n" + "="*50)
    
    # Test a few other issues
    test_cases = [
        'feeling so anxious',
        'very depressed',
        'got promoted today',
        'feeling grateful',
        'hello there'
    ]
    
    for query in test_cases:
        issue = detect_specific_issue(query)
        print(f"'{query}' -> {issue}")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")