# test_sentiment.py
from afa_core.sentiment_analyzer import SentimentAnalyzer
import time

def test_sentiment_analyzer():
    """Test the SentimentAnalyzer functionality including Llama3.2:1b"""
    try:
        # Initialize sentiment analyzer
        print("🔄 Initializing Sentiment Analyzer...")
        analyzer = SentimentAnalyzer()
        print(f"✅ Sentiment Analyzer initialized!")
        print(f"📊 Available models: {analyzer.available_models}")
        print(f"🦙 Llama3.2:1b available: {'llama' in analyzer.available_models}")
        
        # Test sample financial texts
        test_texts = [
            "Apple reports record quarterly revenue, beating analyst expectations",
            "Tesla stock plummets amid production concerns and supply chain issues",
            "Market remains stable with mixed signals from tech sector",
            "Fed announces interest rate cut, boosting investor confidence",
            "Oil prices surge due to geopolitical tensions in the Middle East"
        ]
        
        print("\n🔄 Testing VADER sentiment analysis...")
        for i, text in enumerate(test_texts, 1):
            result = analyzer.get_vader_sentiment(text)
            emoji = "🟢" if result['classification'] == 'Positive' else "🔴" if result['classification'] == 'Negative' else "🟡"
            print(f"   {emoji} Text {i}: {result['classification']} (Score: {result['compound']:.3f})")
            print(f"      \"{text[:60]}...\"")
        
        # Test Llama3.2:1b sentiment if available (FREE - Local LLM)
        if "llama" in analyzer.available_models:
            print(f"\n🦙 Testing Llama3.2:1b sentiment analysis...")
            start_time = time.time()
            
            for i, text in enumerate(test_texts[:3], 1):  # Test first 3 for speed
                result = analyzer.get_llama_sentiment(text)
                if 'error' not in result:
                    emoji = "🟢" if result['classification'] == 'Positive' else "🔴" if result['classification'] == 'Negative' else "🟡"
                    print(f"   {emoji} Text {i}: {result['classification']} (Confidence: {result['confidence']:.3f})")
                    print(f"      Time: {result.get('processing_time', 0):.2f}s | Raw: {result.get('raw_response', 'N/A')[:30]}...")
                else:
                    print(f"   ❌ Error: {result['error']}")
            
            total_time = time.time() - start_time
            print(f"   ⏱️ Total Llama processing time: {total_time:.2f}s")
            
            # Test batch analysis with Llama
            print(f"\n🔄 Testing Llama batch analysis...")
            batch_start = time.time()
            batch_results = analyzer.analyze_text_batch(test_texts[:3], method='llama')
            batch_time = time.time() - batch_start
            
            print(f"   📦 Batch analyzed {len(batch_results)} texts in {batch_time:.2f}s")
            for i, result in enumerate(batch_results, 1):
                if 'error' not in result['sentiment']:
                    sentiment = result['sentiment']['classification']
                    emoji = "🟢" if sentiment == 'Positive' else "🔴" if sentiment == 'Negative' else "🟡"
                    print(f"      {emoji} Batch {i}: {sentiment}")
                else:
                    print(f"      ❌ Batch {i}: {result['sentiment']['error']}")
        else:
            print("\n🦙 Llama3.2:1b not available")
            print("   💡 To enable Llama:")
            print("      1. Install Ollama: https://ollama.com")
            print("      2. Run: ollama pull llama3.2:1b")
            print("      3. Make sure Ollama is running: ollama serve")
        
        # Test FinBERT sentiment if available (FREE - Financial BERT)
        if "finbert" in analyzer.available_models:
            print("\n🔄 Testing FinBERT (Financial BERT) sentiment analysis...")
            result = analyzer.get_finbert_sentiment(test_texts[0])
            if 'error' not in result:
                print(f"   ✅ Classification: {result['classification']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   🤖 Model: {result['model']}")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        # Test Ollama if available (FREE - Local LLM via LangChain)
        if "ollama" in analyzer.available_models and "llama" not in analyzer.available_models:
            print("\n🔄 Testing Ollama (Legacy LangChain) sentiment analysis...")
            result = analyzer.get_ollama_sentiment(test_texts[0])
            if 'error' not in result:
                print(f"   ✅ Classification: {result['classification']}")
                print(f"   🤖 Model: {result['model']}")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        # Test TextBlob (FREE - Simple baseline)
        print("\n🔄 Testing TextBlob sentiment analysis...")
        result = analyzer.get_textblob_sentiment(test_texts[0])
        if 'error' not in result:
            print(f"   ✅ Classification: {result['classification']}")
            print(f"   📊 Polarity: {result['polarity']:.3f}")
            print(f"   📊 Subjectivity: {result['subjectivity']:.3f}")
        else:
            print(f"   ❌ Error: {result['error']}")
        
        # Test consensus sentiment (FREE models only - now includes Llama)
        print("\n🔄 Testing consensus sentiment analysis (FREE models including Llama)...")
        consensus_result = analyzer.get_consensus_sentiment(test_texts[0])
        print(f"   🎯 Consensus: {consensus_result['consensus']}")
        print(f"   📊 Confidence: {consensus_result['confidence']:.3f}")
        print(f"   🤖 Models used: {consensus_result['models_used']}")
        print(f"   📋 Available models: {consensus_result['available_models']}")
        print(f"   🦙 Llama included: {consensus_result.get('llama_available', False)}")
        
        # Show individual model results in consensus
        if 'individual_results' in consensus_result:
            print("   📊 Individual model results:")
            for model, result in consensus_result['individual_results'].items():
                if 'classification' in result:
                    emoji = "🟢" if result['classification'] == 'Positive' else "🔴" if result['classification'] == 'Negative' else "🟡"
                    conf = result.get('confidence', result.get('compound', 0))
                    print(f"      {emoji} {model}: {result['classification']} ({conf:.3f})")
        
        # Test news sentiment for a stock (now uses Llama if available)
        print("\n🔄 Testing news sentiment for AAPL...")
        news_sentiment = analyzer.get_news_sentiment_for_symbol("AAPL")
        
        if 'error' not in news_sentiment:
            print(f"   📈 Overall sentiment for {news_sentiment['company_name']}: {news_sentiment['overall_sentiment']}")
            print(f"   📊 Sentiment score: {news_sentiment['sentiment_score']:.3f}")
            print(f"   📰 Articles analyzed: {news_sentiment['total_articles']}")
            print(f"   🟢 Positive: {news_sentiment['positive_count']}")
            print(f"   🔴 Negative: {news_sentiment['negative_count']}")
            print(f"   🟡 Neutral: {news_sentiment['neutral_count']}")
            print(f"   🤖 Analysis method: {news_sentiment.get('analysis_method', 'consensus')}")
            
            if news_sentiment['news_items']:
                print("\n   📋 Recent news headlines:")
                for item in news_sentiment['news_items']:
                    emoji = "🟢" if item['sentiment'] == 'Positive' else "🔴" if item['sentiment'] == 'Negative' else "🟡"
                    print(f"      {emoji} {item['title'][:80]}...")
        else:
            print(f"   ❌ Error: {news_sentiment['error']}")
        
        # Test batch analysis comparison
        print("\n🔄 Testing batch sentiment analysis (method comparison)...")
        
        methods_to_test = ['consensus']
        if 'llama' in analyzer.available_models:
            methods_to_test.append('llama')
        if 'vader' in analyzer.available_models:
            methods_to_test.append('vader')
        
        for method in methods_to_test:
            print(f"\n   📦 Testing {method.upper()} method:")
            start_time = time.time()
            batch_results = analyzer.analyze_text_batch(test_texts[:3], method=method)
            process_time = time.time() - start_time
            
            print(f"      ⏱️ Processing time: {process_time:.2f}s")
            for i, result in enumerate(batch_results, 1):
                if 'sentiment' in result:
                    if method == 'consensus':
                        sentiment = result['sentiment'].get('consensus', 'Unknown')
                    else:
                        sentiment = result['sentiment'].get('classification', 'Unknown')
                    
                    if sentiment != 'Unknown':
                        emoji = "🟢" if sentiment == 'Positive' else "🔴" if sentiment == 'Negative' else "🟡"
                        print(f"      {emoji} Text {i}: {sentiment}")
                    else:
                        print(f"      ❓ Text {i}: Error or unknown result")
        
        # Performance comparison
        if 'llama' in analyzer.available_models:
            print(f"\n⚡ Performance comparison (single text analysis):")
            test_text = "Apple stock surges on strong quarterly earnings report"
            
            methods = [
                ('VADER', lambda: analyzer.get_vader_sentiment(test_text)),
                ('Llama3.2:1b', lambda: analyzer.get_llama_sentiment(test_text)),
                ('Consensus', lambda: analyzer.get_consensus_sentiment(test_text))
            ]
            
            if 'finbert' in analyzer.available_models:
                methods.append(('FinBERT', lambda: analyzer.get_finbert_sentiment(test_text)))
            
            for method_name, method_func in methods:
                start = time.time()
                try:
                    result = method_func()
                    end = time.time()
                    
                    if 'error' not in result:
                        classification = result.get('classification', result.get('consensus', 'Unknown'))
                        print(f"   {method_name:12}: {classification:8} ({end-start:.3f}s)")
                    else:
                        print(f"   {method_name:12}: Error    ({end-start:.3f}s)")
                except Exception as e:
                    end = time.time()
                    print(f"   {method_name:12}: Failed   ({end-start:.3f}s) - {str(e)[:30]}...")
        
        print("\n🎉 Sentiment analysis testing completed successfully!")
        print("\n💡 FREE Models Available:")
        print("   ✅ VADER - Rule-based sentiment (always available)")
        if 'llama' in analyzer.available_models:
            print("   ✅ Llama3.2:1b - Local LLM sentiment (RECOMMENDED)")
        else:
            print("   💡 Llama3.2:1b - Install Ollama + pull llama3.2:1b")
        print("   ✅ FinBERT - Financial sentiment model (via HuggingFace)")
        print("   ✅ TextBlob - Simple baseline sentiment")
        print("   💡 Ollama - Install for free local LLM: https://ollama.ai/")
        print("   💡 All models work offline after initial download!")
        
        if 'llama' in analyzer.available_models:
            print("\n🦙 Llama3.2:1b is available and will be used as the primary model in consensus!")
            print("   📈 Benefits: Better financial understanding, local processing, no API limits")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sentiment_analyzer()