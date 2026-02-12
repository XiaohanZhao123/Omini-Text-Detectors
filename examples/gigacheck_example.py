"""
GigaCheck Example: AI Text Interval Detection

GigaCheck is a boundary detection method that identifies AI-written character
intervals within mixed human/AI text. Unlike other detectors that classify entire
documents, GigaCheck provides fine-grained character-level segmentation.

Key Features:
- Detects exact character positions of AI-generated content
- Supports mixed human/AI text detection
- Returns intervals: [[start_char, end_char], ...]
- Classification: "human", "ai", or "mixed"
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omini_text import pipeline


def print_result(result, show_intervals=True):
    """Pretty print detection result."""
    print(f"\n{'='*60}")
    print(f"Text: {result['text'][:100]}...")
    print(f"Label: {'AI' if result['label'] == 1 else 'Human'}")
    print(f"Score: {result['score']:.3f}")
    print(f"Prediction: {result['metadata']['pred_label']}")
    
    if show_intervals and result['metadata']['ai_intervals']:
        print(f"\nAI Intervals (character positions):")
        for i, interval in enumerate(result['metadata']['ai_intervals']):
            start, end = interval[0], interval[1]
            text_snippet = result['text'][start:end]
            print(f"  Interval {i+1}: [{start}, {end}]")
            print(f"    Text: \"{text_snippet}\"")
            if len(interval) > 2:
                print(f"    Confidence: {interval[2]:.3f}")
    print(f"{'='*60}\n")


def visualize_intervals(text, intervals):
    """Visualize AI intervals in text with markers."""
    if not intervals:
        print("No AI intervals detected.")
        return
    
    # Create a visualization with markers
    markers = [' '] * len(text)
    for interval in intervals:
        start, end = interval[0], interval[1]
        for i in range(start, min(end, len(text))):
            markers[i] = '^' if i == start or i == end-1 else '='
    
    print("\nVisualization (^ marks interval boundaries, = marks AI content):")
    print(text)
    print(''.join(markers))
    print()


def example_basic_usage():
    """Example 1: Basic usage with AI-generated text."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage - Pure AI Text")
    print("="*60)
    
    # Initialize pipeline
    pipe = pipeline("ai-text-detection", model="gigacheck", device="cuda:0")
    
    # AI-generated text (typical GPT-style)
    ai_text = """
    Artificial intelligence has revolutionized numerous industries by automating 
    complex tasks and enabling data-driven decision making. Machine learning 
    algorithms can now process vast amounts of information to identify patterns 
    that would be impossible for humans to detect manually.
    """
    
    result = pipe(ai_text)
    print_result(result)


def example_mixed_text():
    """Example 2: Mixed human/AI text detection."""
    print("\n" + "="*60)
    print("Example 2: Mixed Human/AI Text")
    print("="*60)
    
    pipe = pipeline("ai-text-detection", model="gigacheck", device="cuda:0")
    
    # Mixed text: human intro, AI middle, human ending
    mixed_text = (
        "I've been thinking about this problem for a while. "
        "Machine learning algorithms leverage statistical patterns in data "
        "to make predictions and classifications without explicit programming. "
        "The key advantage is their ability to generalize from training examples. "
        "But honestly, I'm not sure if this approach will work in our case."
    )
    
    result = pipe(mixed_text)
    print_result(result)
    visualize_intervals(mixed_text, result['metadata']['ai_intervals'])


def example_human_text():
    """Example 3: Human-written text (should have no AI intervals)."""
    print("\n" + "="*60)
    print("Example 3: Human-Written Text")
    print("="*60)
    
    pipe = pipeline("ai-text-detection", model="gigacheck", device="cuda:0")
    
    # Human-written text (casual, personal)
    human_text = (
        "So I was walking down the street yesterday and ran into my old friend. "
        "We haven't seen each other in like three years! It was crazy how much "
        "things have changed. We ended up grabbing coffee and catching up for hours."
    )
    
    result = pipe(human_text)
    print_result(result)
    
    if result['metadata']['ai_intervals']:
        visualize_intervals(human_text, result['metadata']['ai_intervals'])
    else:
        print("No AI intervals detected - text appears to be human-written.")


def example_batch_processing():
    """Example 4: Batch processing multiple texts."""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    pipe = pipeline("ai-text-detection", model="gigacheck", device="cuda:0")
    
    texts = [
        "This is a simple human sentence.",
        "The utilization of advanced neural network architectures enables "
        "sophisticated pattern recognition capabilities.",
        "Hey, what's up? Just checking in to see how you're doing.",
    ]
    
    results = pipe(texts)
    
    for i, result in enumerate(results):
        print(f"\nText {i+1}:")
        print_result(result, show_intervals=False)


def example_interval_analysis():
    """Example 5: Detailed interval analysis."""
    print("\n" + "="*60)
    print("Example 5: Detailed Interval Analysis")
    print("="*60)
    
    pipe = pipeline("ai-text-detection", model="gigacheck", device="cuda:0")
    
    # Complex mixed text
    complex_text = (
        "Personal note: I've been working on this project. "
        "The implementation leverages transformer-based architectures "
        "to achieve state-of-the-art performance on benchmark datasets. "
        "However, I'm concerned about the computational requirements. "
        "We might need to optimize the inference pipeline."
    )
    
    result = pipe(complex_text)
    
    print(f"Full text: {complex_text}\n")
    print(f"Classification: {result['metadata']['pred_label']}")
    print(f"AI Score: {result['score']:.3f}")
    
    intervals = result['metadata']['ai_intervals']
    if intervals:
        print(f"\nFound {len(intervals)} AI interval(s):")
        total_ai_chars = 0
        for i, interval in enumerate(intervals):
            start, end = interval[0], interval[1]
            length = end - start
            total_ai_chars += length
            percentage = (length / len(complex_text)) * 100
            print(f"\n  Interval {i+1}:")
            print(f"    Position: [{start}, {end}]")
            print(f"    Length: {length} characters ({percentage:.1f}% of text)")
            print(f"    Content: \"{complex_text[start:end]}\"")
        
        overall_percentage = (total_ai_chars / len(complex_text)) * 100
        print(f"\n  Total AI content: {total_ai_chars}/{len(complex_text)} chars ({overall_percentage:.1f}%)")
    else:
        print("\nNo AI intervals detected.")
    
    visualize_intervals(complex_text, intervals)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("GigaCheck Interval Detection Examples")
    print("="*60)
    print("\nGigaCheck detects AI-written character intervals in text.")
    print("It can identify mixed human/AI content with precise boundaries.\n")
    
    try:
        # Run examples
        example_basic_usage()
        example_mixed_text()
        example_human_text()
        example_batch_processing()
        example_interval_analysis()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            pipe.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()

