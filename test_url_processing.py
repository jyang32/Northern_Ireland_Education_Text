#!/usr/bin/env python3
"""
Test script for URL processing functionality.
"""

import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent / "scripts"))

from utils import preprocess_combined, extract_urls_from_text, fetch_url_content

def test_url_extraction():
    """Test URL extraction from text."""
    print("Testing URL extraction...")
    
    test_text = """
    Here is some text with URLs:
    Check out https://en.wikipedia.org/wiki/Northern_Ireland for more info.
    Also visit https://www.bbc.com/news/uk-northern-ireland for news.
    And https://www.gov.uk/government/organisations/northern-ireland-office for official info.
    """
    
    urls = extract_urls_from_text(test_text)
    print(f"Found {len(urls)} URLs:")
    for url in urls:
        print(f"  - {url}")
    
    return urls

def test_url_fetching():
    """Test fetching content from a single URL."""
    print("\nTesting URL content fetching...")
    
    # Test with a simple, reliable URL
    test_url = "https://httpbin.org/html"
    
    print(f"Fetching content from: {test_url}")
    content = fetch_url_content(test_url, max_chars=1000)
    
    print(f"Content length: {len(content)} characters")
    print(f"First 200 characters: {content[:200]}...")
    
    return content

def test_preprocess_combined():
    """Test the complete preprocess_combined function."""
    print("\nTesting preprocess_combined function...")
    
    test_text = """
    This is a combined document with URLs:
    
    For information about Northern Ireland, visit:
    https://en.wikipedia.org/wiki/Northern_Ireland
    
    For educational resources, check:
    https://www.htani.org/gcse-ni-troubles-resources/
    
    This document contains Q&A sections and other content.
    """
    
    print("Original text length:", len(test_text))
    
    # Test with URL fetching enabled
    processed_text, url_contents_dict = preprocess_combined(test_text, fetch_urls=True, max_url_chars=2000)
    print("Processed text length (with URLs):", len(processed_text))
    print("URL contents found:", len(url_contents_dict))
    print("First 500 characters of processed text:")
    print(processed_text[:500])
    
    # Test with URL fetching disabled
    processed_text_no_urls, url_contents_dict_no = preprocess_combined(test_text, fetch_urls=False)
    print("\nProcessed text length (without URLs):", len(processed_text_no_urls))
    print("URL contents found (disabled):", len(url_contents_dict_no))
    
    return processed_text

if __name__ == "__main__":
    print("=== URL Processing Test ===\n")
    
    try:
        # Test URL extraction
        urls = test_url_extraction()
        
        # Test URL fetching (only if we have internet)
        try:
            content = test_url_fetching()
        except Exception as e:
            print(f"URL fetching test failed (likely no internet): {e}")
        
        # Test complete function
        processed = test_preprocess_combined()
        
        print("\n=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 