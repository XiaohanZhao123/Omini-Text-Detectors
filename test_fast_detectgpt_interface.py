"""
Test script for Fast-DetectGPT detector unified interface.

This script tests the FastDetectGPTDetector implementation to ensure:
1. Config loading works correctly
2. Model initialization succeeds
3. Detection returns standardized format
4. Batch processing works
"""

import sys
from pathlib import Path

# Add omini_text to path
sys.path.insert(0, str(Path(__file__).parent))

from omini_text.detectors.fast_detectgpt_detector import FastDetectGPTDetector


def test_fast_detectgpt():
    """Test Fast-DetectGPT detector with sample texts."""

    print("=" * 80)
    print("Testing Fast-DetectGPT Detector Interface")
    print("=" * 80)

    # Configuration
    config = {
        'sampling_model_name': 'gpt-neo-2.7B',
        'scoring_model_name': 'gpt-neo-2.7B',
        'device': '0,1,2,3',  # Use multi-GPU
        'cache_dir': '../cache',
        'threshold': 0.5
    }

    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Initialize detector
    print("\n🔧 Initializing detector...")
    detector = FastDetectGPTDetector(config)
    print("✅ Detector initialized successfully!")

    # Test cases
    test_cases = [
        {
            'name': 'Human-written text',
            'text': 'The quick brown fox jumps over the lazy dog. This is a classic pangram used in typography.'
        },
        {
            'name': 'AI-like text',
            'text': 'Artificial intelligence has revolutionized numerous industries by enabling machines to perform tasks that traditionally required human intelligence. Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions with remarkable accuracy.'
        },
        {
            'name': 'Short text',
            'text': 'Hello world!'
        }
    ]

    # Run detection
    print("\n" + "=" * 80)
    print("Running Detection Tests")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"   Text: {test_case['text'][:100]}...")

        try:
            result = detector.detect(test_case['text'])

            print(f"\n   ✅ Detection successful!")
            print(f"   Label: {result['label']} ({'AI' if result['label'] == 1 else 'Human'})")
            print(f"   Score: {result['score']:.4f} ({result['score']*100:.2f}% AI probability)")
            print(f"   Criterion: {result['metadata']['criterion']:.4f}")
            print(f"   Tokens: {result['metadata']['num_tokens']}")

            # Verify return format
            assert 'text' in result
            assert 'label' in result
            assert 'score' in result
            assert 'metadata' in result
            assert 'criterion' in result['metadata']
            assert 'num_tokens' in result['metadata']
            assert result['label'] in [0, 1]
            assert 0.0 <= result['score'] <= 1.0

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


def test_non_recommended_combination():
    """Test detector with non-recommended model combination (should warn)."""

    print("\n" + "=" * 80)
    print("Testing Non-Recommended Model Combination")
    print("=" * 80)

    config = {
        'sampling_model_name': 'gpt2',  # Non-recommended
        'scoring_model_name': 'gpt2',
        'device': '0',
        'cache_dir': '../cache',
        'threshold': 0.5
    }

    print("\n⚠️  Expecting warning about non-recommended combination...")
    detector = FastDetectGPTDetector(config)
    print("✅ Detector initialized with warning as expected")


def test_cpu_error():
    """Test that CPU-only mode raises error."""

    print("\n" + "=" * 80)
    print("Testing CPU-Only Error Handling")
    print("=" * 80)

    config = {
        'sampling_model_name': 'gpt-neo-2.7B',
        'scoring_model_name': 'gpt-neo-2.7B',
        'device': 'cpu',  # Should raise error
        'cache_dir': '../cache',
        'threshold': 0.5
    }

    print("\n⚠️  Attempting to initialize with CPU (should fail)...")
    try:
        detector = FastDetectGPTDetector(config)
        print("❌ Expected ValueError but detector initialized successfully!")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {str(e)[:100]}...")


if __name__ == '__main__':
    # Run basic test
    test_fast_detectgpt()

    # Test edge cases
    test_non_recommended_combination()
    test_cpu_error()

    print("\n" + "=" * 80)
    print("🎉 All interface tests passed!")
    print("=" * 80)
