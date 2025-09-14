# afa_core/sentiment_analyzer.py

import os
import numpy as np
import requests
import yfinance as yf
import json
import time
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from typing import Dict, List, Optional

# LangChain imports for free alternatives
try:
    from langchain_community.llms import Ollama
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not fully available. Some features may be limited.")

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis for financial data using multiple approaches including Llama3.2:1b
    """
    
    def __init__(self, llama_model="llama3.2:1b", ollama_base_url="http://localhost:11434"):
        """Initialize sentiment analyzer with free alternatives including Llama"""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Llama configuration
        self.llama_model = llama_model
        self.ollama_base_url = ollama_base_url
        self.ollama_api_url = f"{ollama_base_url}/api/generate"
        
        # Initialize available models (free alternatives only)
        self.available_models = []
        self._check_available_models()
        
        # Initialize HuggingFace models
        self.finbert_pipeline = None
        self.general_sentiment_pipeline = None
        self._initialize_hf_models()
    
    def _check_available_models(self):
        """Check which free sentiment analysis models are available"""
        # VADER is always available
        self.available_models.append("vader")
        
        # Check for transformers/HuggingFace (free)
        try:
            from transformers import pipeline
            self.available_models.append("huggingface")
            self.available_models.append("finbert")  # Financial BERT
        except ImportError:
            print("⚠️ Install transformers for free models: pip install transformers torch")
        
        # Check for Ollama with Llama (free local LLM)
        if self._check_ollama_llama_status():
            self.available_models.append("llama")
            self.available_models.append("ollama")
        
        # Check for legacy Ollama support
        if LANGCHAIN_AVAILABLE:
            try:
                from langchain_community.llms import Ollama
                if "llama" not in self.available_models:
                    self.available_models.append("ollama")
            except Exception:
                print("💡 Install Ollama for free local LLM: https://ollama.ai/")
    
    def _check_ollama_llama_status(self) -> bool:
        """Check if Ollama is running and Llama model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
                
            # Check if our Llama model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.llama_model not in model_names:
                print(f"💡 Llama model {self.llama_model} not found. Run: ollama pull {self.llama_model}")
                return False
                
            print(f"✅ Ollama is running with {self.llama_model}")
            return True
            
        except requests.exceptions.RequestException:
            print("💡 Ollama not running. Start with: ollama serve")
            return False
    
    def _initialize_hf_models(self):
        """Initialize HuggingFace models for financial sentiment"""
        try:
            from transformers import pipeline
            
            # Try different models in order of preference
            models_to_try = [
                {
                    "name": "FinBERT",
                    "model": "ProsusAI/finbert",
                    "tokenizer": "ProsusAI/finbert"
                },
                {
                    "name": "DistilBERT-Financial",
                    "model": "distilbert-base-uncased-finetuned-sst-2-english"
                },
                {
                    "name": "Default-Sentiment",
                    "model": None  # Use default
                }
            ]
            
            for model_config in models_to_try:
                try:
                    print(f"🔄 Loading {model_config['name']} model...")
                    
                    if model_config['model']:
                        if 'tokenizer' in model_config:
                            self.finbert_pipeline = pipeline(
                                "sentiment-analysis",
                                model=model_config['model'],
                                tokenizer=model_config['tokenizer'],
                                framework="pt"  # Use PyTorch instead of TensorFlow
                            )
                        else:
                            self.finbert_pipeline = pipeline(
                                "sentiment-analysis",
                                model=model_config['model'],
                                framework="pt"
                            )
                    else:
                        self.finbert_pipeline = pipeline("sentiment-analysis")
                    
                    print(f"✅ {model_config['name']} loaded successfully!")
                    break
                    
                except Exception as model_error:
                    print(f"⚠️ Failed to load {model_config['name']}: {model_error}")
                    continue
            
            # Try to load a general backup model
            if self.finbert_pipeline is None:
                try:
                    self.general_sentiment_pipeline = pipeline("sentiment-analysis")
                    print("✅ Default sentiment pipeline loaded as backup")
                except Exception as e:
                    print(f"⚠️ Could not load any HuggingFace models: {e}")
            
        except ImportError:
            print("⚠️ Transformers not available. Install with: pip install transformers torch")
        except Exception as e:
            print(f"⚠️ Error initializing HuggingFace models: {e}")
    
    def _create_financial_prompt(self, text: str) -> str:
        """Create optimized prompt for financial sentiment analysis with Llama"""
        return f"""You are a financial analyst. Analyze the sentiment of this financial text and respond with ONLY ONE WORD.

Rules:
- POSITIVE: Good news, gains, growth, optimism, bullish signals
- NEGATIVE: Bad news, losses, decline, pessimism, bearish signals  
- NEUTRAL: Factual statements, mixed signals, no clear direction

Text: "{text}"

Sentiment:"""

    def _extract_sentiment_from_llama(self, response_text: str) -> str:
        """Extract and normalize sentiment from Llama response"""
        response = response_text.strip().upper()
        
        # Look for sentiment words
        if any(word in response for word in ['POSITIVE', 'BULLISH', 'GOOD', 'UP', 'GAIN', 'RISE']):
            return 'Positive'
        elif any(word in response for word in ['NEGATIVE', 'BEARISH', 'BAD', 'DOWN', 'LOSS', 'FALL']):
            return 'Negative'
        elif 'NEUTRAL' in response:
            return 'Neutral'
        else:
            # Try to parse the first word
            words = response.split()
            if words:
                first_word = words[0].strip('.,!?')
                if first_word in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                    return first_word.capitalize()
                    
        return 'Neutral'  # Default fallback
    
    def get_vader_sentiment(self, text):
        """
        Get sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores including compound score
        """
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'classification': self._classify_vader_score(scores['compound'])
        }
    
    def _classify_vader_score(self, compound_score):
        """Convert VADER compound score to classification"""
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def get_llama_sentiment(self, text, timeout=10):
        """
        Get sentiment using Llama3.2:1b via Ollama - FREE and LOCAL
        
        Args:
            text (str): Text to analyze
            timeout (int): Request timeout in seconds
            
        Returns:
            dict: Sentiment analysis results
        """
        if "llama" not in self.available_models:
            return {"error": "Llama model not available. Check if Ollama is running and model is pulled."}
        
        if not text or len(text.strip()) == 0:
            return {
                'classification': 'Neutral',
                'confidence': 0.0,
                'model': self.llama_model
            }
        
        start_time = time.time()
        
        try:
            prompt = self._create_financial_prompt(text)
            
            payload = {
                'model': self.llama_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,  # Low temperature for consistent results
                    'top_p': 0.9,
                    'max_tokens': 10     # We only need one word
                }
            }
            
            response = requests.post(
                self.ollama_api_url,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}")
            
            result = response.json()
            llama_response = result.get('response', '').strip()
            
            classification = self._extract_sentiment_from_llama(llama_response)
            processing_time = time.time() - start_time
            
            # Calculate confidence based on response clarity
            confidence = 0.8 if classification != 'Neutral' else 0.6
            if any(word in llama_response.upper() for word in ['VERY', 'STRONG', 'CLEAR']):
                confidence = min(confidence + 0.1, 0.95)
            
            return {
                'classification': classification,
                'confidence': confidence,
                'model': self.llama_model,
                'processing_time': processing_time,
                'raw_response': llama_response
            }
            
        except requests.exceptions.Timeout:
            return {
                'classification': 'Neutral',
                'confidence': 0.0,
                'model': f"{self.llama_model}_timeout",
                'error': 'timeout'
            }
            
        except Exception as e:
            return {
                'classification': 'Neutral',
                'confidence': 0.0,
                'model': f"{self.llama_model}_error",
                'error': str(e)
            }
    
    def get_finbert_sentiment(self, text):
        """
        Get sentiment using FinBERT (Financial BERT) - FREE
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if self.finbert_pipeline is None:
            return {"error": "FinBERT model not available"}
        
        try:
            result = self.finbert_pipeline(text)[0]
            
            # FinBERT returns: positive, negative, neutral
            label = result['label'].lower()
            if label == 'positive':
                classification = "Positive"
            elif label == 'negative':
                classification = "Negative"
            else:
                classification = "Neutral"
            
            return {
                'classification': classification,
                'confidence': result['score'],
                'model': 'FinBERT',
                'raw_result': result
            }
            
        except Exception as e:
            return {"error": f"FinBERT error: {str(e)}"}
    
    def get_huggingface_sentiment(self, text):
        """
        Get sentiment using HuggingFace transformers - FREE
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if "huggingface" not in self.available_models:
            return {"error": "HuggingFace transformers not available"}
        
        try:
            # Try FinBERT first for financial texts
            if self.finbert_pipeline:
                return self.get_finbert_sentiment(text)
            
            # Fallback to general sentiment
            if self.general_sentiment_pipeline:
                result = self.general_sentiment_pipeline(text)[0]
                
                # Normalize labels
                label = result['label'].upper()
                if label in ['POSITIVE', 'POS', 'LABEL_2']:
                    classification = "Positive"
                elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
                    classification = "Negative"
                else:
                    classification = "Neutral"
                
                return {
                    'classification': classification,
                    'confidence': result['score'],
                    'model': 'HuggingFace-General',
                    'raw_result': result
                }
            
            return {"error": "No HuggingFace models available"}
            
        except Exception as e:
            return {"error": f"HuggingFace error: {str(e)}"}
    
    def get_ollama_sentiment(self, text, model="llama2"):
        """
        Get sentiment using legacy Ollama (LangChain) - FREE
        
        Args:
            text (str): Text to analyze
            model (str): Ollama model to use
            
        Returns:
            dict: Sentiment analysis results
        """
        if "ollama" not in self.available_models:
            return {"error": "Ollama not available"}
        
        try:
            from langchain_community.llms import Ollama
            
            llm = Ollama(model=model, temperature=0)
            
            prompt = f"""Analyze the sentiment of this financial text and respond with only one word: Positive, Negative, or Neutral.

Text: "{text}"

Sentiment:"""
            
            response = llm.invoke(prompt)
            classification = response.strip().split()[0]  # Get first word
            
            # Normalize the response
            if classification.lower() in ['positive', 'bullish', 'good']:
                classification = "Positive"
            elif classification.lower() in ['negative', 'bearish', 'bad']:
                classification = "Negative"
            else:
                classification = "Neutral"
            
            return {
                'classification': classification,
                'model': f'Ollama-{model}',
                'raw_response': response
            }
            
        except Exception as e:
            return {"error": f"Ollama error: {str(e)}"}
    
    def get_textblob_sentiment(self, text):
        """
        Get sentiment using TextBlob - FREE (simple baseline)
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                classification = "Positive"
            elif polarity < -0.1:
                classification = "Negative"
            else:
                classification = "Neutral"
            
            return {
                'classification': classification,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'model': 'TextBlob'
            }
            
        except ImportError:
            return {"error": "TextBlob not available. Install with: pip install textblob"}
        except Exception as e:
            return {"error": f"TextBlob error: {str(e)}"}
    
    def get_consensus_sentiment(self, text):
        """
        Get consensus sentiment from multiple FREE models including Llama3.2:1b
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Consensus sentiment analysis
        """
        results = {}
        
        # Get VADER sentiment (always available)
        results['vader'] = self.get_vader_sentiment(text)
        
        # Get Llama sentiment if available (PREFERRED for financial analysis)
        if "llama" in self.available_models:
            results['llama'] = self.get_llama_sentiment(text)
        
        # Get FinBERT sentiment if available (FREE)
        if "finbert" in self.available_models:
            results['finbert'] = self.get_finbert_sentiment(text)
        
        # Get HuggingFace sentiment if available (FREE)
        if "huggingface" in self.available_models and 'finbert' not in results:
            results['huggingface'] = self.get_huggingface_sentiment(text)
        
        # Get Ollama sentiment if available (FREE) - only if Llama not available
        if "ollama" in self.available_models and 'llama' not in results:
            results['ollama'] = self.get_ollama_sentiment(text)
        
        # Try TextBlob as additional free option
        textblob_result = self.get_textblob_sentiment(text)
        if 'error' not in textblob_result:
            results['textblob'] = textblob_result
        
        # Calculate consensus with higher weight for Llama and FinBERT
        classifications = []
        confidences = []
        weights = []
        
        for model, result in results.items():
            if 'classification' in result:
                classifications.append(result['classification'])
                
                # Get confidence if available
                if 'confidence' in result:
                    confidence = result['confidence']
                elif 'compound' in result:  # VADER
                    confidence = abs(result['compound'])
                else:
                    confidence = 0.5  # Default confidence
                
                confidences.append(confidence)
                
                # Assign weights - Llama and FinBERT get higher weights
                if model == 'llama':
                    weights.append(1.5)  # Highest weight for Llama
                elif model == 'finbert':
                    weights.append(1.3)  # High weight for FinBERT
                elif model == 'huggingface':
                    weights.append(1.0)
                else:
                    weights.append(0.8)  # Lower weight for other methods
        
        # Weighted majority vote
        if classifications:
            from collections import defaultdict
            weighted_votes = defaultdict(float)
            
            for classification, confidence, weight in zip(classifications, confidences, weights):
                weighted_votes[classification] += confidence * weight
            
            # Get the classification with highest weighted vote
            consensus = max(weighted_votes, key=weighted_votes.get)
            total_weight = sum(weighted_votes.values())
            confidence = weighted_votes[consensus] / total_weight if total_weight > 0 else 0.0
        else:
            consensus = "Neutral"
            confidence = 0.0
        
        return {
            'consensus': consensus,
            'confidence': confidence,
            'individual_results': results,
            'models_used': len([r for r in results.values() if 'classification' in r]),
            'available_models': self.available_models,
            'llama_available': 'llama' in self.available_models
        }
    
    def get_news_sentiment_for_symbol(self, symbol, days_back=7):
        """
        Get news sentiment for a specific stock symbol using the best available method
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            days_back (int): Number of days to look back for news
            
        Returns:
            dict: Aggregated sentiment analysis of recent news
        """
        try:
            # Get company name from yfinance
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)
            
            # Get news (using a simple approach with yfinance news)
            news_data = ticker.news
            
            if not news_data:
                return {"error": "No news data available"}
            
            # Analyze sentiment for each news item using the best available method
            sentiments = []
            news_items = []
            
            for item in news_data[:10]:  # Limit to recent 10 items
                title = item.get('title', '')
                if title:
                    # Use Llama if available, otherwise use consensus
                    if "llama" in self.available_models:
                        sentiment_result = self.get_llama_sentiment(title)
                        if 'error' not in sentiment_result:
                            sentiment_result = {'consensus': sentiment_result['classification'], **sentiment_result}
                        else:
                            sentiment_result = self.get_consensus_sentiment(title)
                    else:
                        sentiment_result = self.get_consensus_sentiment(title)
                    
                    sentiments.append(sentiment_result)
                    news_items.append({
                        'title': title,
                        'sentiment': sentiment_result.get('consensus', sentiment_result.get('classification', 'Neutral')),
                        'url': item.get('link', ''),
                        'published': item.get('providerPublishTime', 0)
                    })
            
            # Calculate overall sentiment
            if sentiments:
                positive_count = sum(1 for s in sentiments 
                                   if s.get('consensus', s.get('classification', 'Neutral')) == 'Positive')
                negative_count = sum(1 for s in sentiments 
                                   if s.get('consensus', s.get('classification', 'Neutral')) == 'Negative')
                neutral_count = sum(1 for s in sentiments 
                                  if s.get('consensus', s.get('classification', 'Neutral')) == 'Neutral')
                
                total = len(sentiments)
                sentiment_score = (positive_count - negative_count) / total
                
                if sentiment_score > 0.2:
                    overall_sentiment = "Positive"
                elif sentiment_score < -0.2:
                    overall_sentiment = "Negative"
                else:
                    overall_sentiment = "Neutral"
            else:
                overall_sentiment = "Neutral"
                sentiment_score = 0.0
                positive_count = negative_count = neutral_count = 0
            
            return {
                'symbol': symbol,
                'company_name': company_name,
                'overall_sentiment': overall_sentiment,
                'sentiment_score': sentiment_score,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'total_articles': len(sentiments),
                'news_items': news_items[:5],  # Return top 5 for reference
                'analysis_method': 'llama' if 'llama' in self.available_models else 'consensus'
            }
            
        except Exception as e:
            return {"error": f"Error analyzing news sentiment: {str(e)}"}
    
    def analyze_text_batch(self, texts, method='consensus'):
        """
        Analyze sentiment for multiple texts using FREE models including Llama
        
        Args:
            texts (list): List of texts to analyze
            method (str): Method to use ('vader', 'llama', 'finbert', 'huggingface', 'ollama', 'textblob', 'consensus')
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        
        # If using Llama and have multiple texts, we can optimize with batch processing
        if method == 'llama' and len(texts) <= 5 and 'llama' in self.available_models:
            return self._batch_analyze_llama(texts)
        
        for text in texts:
            if method == 'vader':
                result = self.get_vader_sentiment(text)
            elif method == 'llama':
                result = self.get_llama_sentiment(text)
            elif method == 'finbert':
                result = self.get_finbert_sentiment(text)
            elif method == 'huggingface':
                result = self.get_huggingface_sentiment(text)
            elif method == 'ollama':
                result = self.get_ollama_sentiment(text)
            elif method == 'textblob':
                result = self.get_textblob_sentiment(text)
            elif method == 'consensus':
                result = self.get_consensus_sentiment(text)
            else:
                result = {"error": f"Unknown method: {method}. Available: {self.available_models}"}
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': result
            })
        
        return results
    
    def _batch_analyze_llama(self, texts: List[str]) -> List[Dict]:
        """Optimized batch analysis using Llama for multiple texts"""
        if "llama" not in self.available_models or not texts:
            return []
        
        try:
            # Create batch prompt
            text_list = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
            
            prompt = f"""Analyze the sentiment of each financial text below. Respond with only the sentiment for each line (POSITIVE, NEGATIVE, or NEUTRAL).

Format your response exactly like this:
1. POSITIVE
2. NEGATIVE
3. NEUTRAL

Texts:
{text_list}

Sentiments:"""
            
            payload = {
                'model': self.llama_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'max_tokens': len(texts) * 5
                }
            }
            
            response = requests.post(self.ollama_api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                llama_response = result.get('response', '').strip()
                
                # Parse batch response
                lines = llama_response.split('\n')
                sentiments = []
                
                for line in lines:
                    if line.strip():
                        sentiment = self._extract_sentiment_from_llama(line)
                        sentiments.append(sentiment)
                
                # Create results
                results = []
                for i, text in enumerate(texts):
                    sentiment = sentiments[i] if i < len(sentiments) else 'Neutral'
                    confidence = 0.8 if sentiment != 'Neutral' else 0.6
                    
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': {
                            'classification': sentiment,
                            'confidence': confidence,
                            'model': f"{self.llama_model}_batch"
                        }
                    })
                
                return results
                
        except Exception as e:
            print(f"Batch Llama analysis failed: {e}, falling back to individual analysis")
        
        # Fallback to individual analysis
        return [{'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': self.get_llama_sentiment(text)} for text in texts]