# afa_core/sentiment_analyzer.py

import os
import logging
import numpy as np
import requests
import yfinance as yf
import json
import time
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain imports for free alternatives
LANGCHAIN_AVAILABLE = False
try:
    from langchain_community.llms import Ollama
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not fully available. Some features may be limited. Install with: pip install langchain-community langchain-huggingface transformers torch")

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
            logger.warning("Install transformers for FinBERT/HuggingFace models: pip install transformers torch")

        # Check for Ollama with Llama (free local LLM) - direct API
        if self._check_ollama_llama_status():
            self.available_models.append("llama") # Represents the direct API call using self.llama_model

        # Check for legacy Ollama support via LangChain
        if LANGCHAIN_AVAILABLE:
            try:
                from langchain_community.llms import Ollama
                # Only add if direct llama API isn't already the primary "ollama" handler
                if "llama" not in self.available_models:
                    self.available_models.append("ollama") # Represents LangChain-based Ollama
            except Exception:
                logger.info("LangChain Ollama not fully functional. Make sure Ollama is installed and running: https://ollama.ai/")


    def _check_ollama_llama_status(self) -> bool:
        """Check if Ollama is running and Llama model is available via direct API"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.info(f"Ollama server not reachable at {self.ollama_base_url}. Status code: {response.status_code}")
                return False

            # Check if our Llama model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]

            if self.llama_model not in model_names:
                logger.info(f"Llama model '{self.llama_model}' not found on Ollama server. Run: ollama pull {self.llama_model}")
                return False

            logger.info(f"Ollama is running with '{self.llama_model}' available.")
            return True

        except requests.exceptions.ConnectionError:
            logger.info(f"Ollama server not running or unreachable at {self.ollama_base_url}. Start with: ollama serve")
            return False
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama server connection timed out at {self.ollama_base_url}.")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return False

    def _initialize_hf_models(self):
        """Initialize HuggingFace models for financial sentiment"""
        if "huggingface" not in self.available_models:
            logger.info("HuggingFace models skipped as transformers are not available.")
            return

        try:
            from transformers import pipeline

            models_to_try = [
                {
                    "name": "FinBERT",
                    "model": "ProsusAI/finbert",
                    "tokenizer": "ProsusAI/finbert"
                },
                {
                    "name": "DistilBERT-Financial", # A common general sentiment model
                    "model": "distilbert-base-uncased-finetuned-sst-2-english"
                },
                {
                    "name": "Default-Sentiment",
                    "model": None # Use default pipeline model
                }
            ]

            for model_config in models_to_try:
                try:
                    logger.info(f"Loading {model_config['name']} model...")

                    if model_config['model']:
                        if 'tokenizer' in model_config:
                            self.finbert_pipeline = pipeline(
                                "sentiment-analysis",
                                model=model_config['model'],
                                tokenizer=model_config['tokenizer'],
                                framework="pt"  # Use PyTorch
                            )
                        else:
                            self.finbert_pipeline = pipeline(
                                "sentiment-analysis",
                                model=model_config['model'],
                                framework="pt"
                            )
                    else:
                        self.finbert_pipeline = pipeline("sentiment-analysis") # Default general model

                    logger.info(f"✅ {model_config['name']} loaded successfully!")
                    # If FinBERT is loaded, it's the primary financial pipeline
                    if model_config['name'] == "FinBERT":
                        break # Stop trying other models if FinBERT is loaded
                    # If not FinBERT but a financial-specific model was loaded, keep it as general
                    elif self.general_sentiment_pipeline is None and model_config['name'] == "DistilBERT-Financial":
                        self.general_sentiment_pipeline = self.finbert_pipeline # Use it as the general fallback
                        self.finbert_pipeline = None # Reset finbert_pipeline as it's not the specific FinBERT model
                        logger.info("Using DistilBERT-Financial as a general sentiment fallback.")
                        break
                    elif self.general_sentiment_pipeline is None and model_config['name'] == "Default-Sentiment":
                        self.general_sentiment_pipeline = self.finbert_pipeline
                        self.finbert_pipeline = None
                        logger.info("Using default HuggingFace sentiment pipeline as a general fallback.")
                        break

                except Exception as model_error:
                    logger.warning(f"Failed to load {model_config['name']}: {model_error}")
                    continue

            # Ensure at least a general pipeline is tried if finbert wasn't specifically loaded
            if self.finbert_pipeline is None and self.general_sentiment_pipeline is None:
                try:
                    self.general_sentiment_pipeline = pipeline("sentiment-analysis")
                    logger.info("✅ Default sentiment pipeline loaded as backup.")
                except Exception as e:
                    logger.warning(f"Could not load any HuggingFace models, including general backup: {e}")

        except ImportError:
            # This case should be caught by the _check_available_models already
            logger.warning("Transformers not available for HuggingFace models.")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace models: {e}")

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
        Get sentiment using Llama3.2:1b via Ollama - FREE and LOCAL (Direct API)

        Args:
            text (str): Text to analyze
            timeout (int): Request timeout in seconds

        Returns:
            dict: Sentiment analysis results
        """
        if "llama" not in self.available_models:
            return {"error": "Llama model (direct API) not available. Check if Ollama is running and model is pulled.", "classification": "Neutral", "confidence": 0.0, "model": "None"}

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
                    'num_predict': 10     # We only need a few tokens
                }
            }

            response = requests.post(
                self.ollama_api_url,
                json=payload,
                timeout=timeout
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API request failed with status: {response.status_code}, response: {response.text}")

            result = response.json()
            llama_response = result.get('response', '').strip()

            classification = self._extract_sentiment_from_llama(llama_response)
            processing_time = time.time() - start_time

            # Calculate confidence based on response clarity
            # Simple heuristic: higher confidence for clear Positive/Negative, lower for Neutral
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
            logger.warning(f"Llama sentiment analysis timed out for text: {text[:50]}...")
            return {
                'classification': 'Neutral',
                'confidence': 0.0,
                'model': f"{self.llama_model}_timeout",
                'error': 'timeout'
            }
        except requests.exceptions.ConnectionError:
            logger.error(f"Llama sentiment analysis failed due to connection error. Is Ollama server running?")
            return {
                'classification': 'Neutral',
                'confidence': 0.0,
                'model': f"{self.llama_model}_connection_error",
                'error': 'connection error'
            }
        except Exception as e:
            logger.error(f"Error getting Llama sentiment for text '{text[:50]}...': {e}")
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
            return {"error": "FinBERT model not available", "classification": "Neutral", "confidence": 0.0, "model": "None"}

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
            logger.error(f"FinBERT error for text '{text[:50]}...': {str(e)}")
            return {"error": f"FinBERT error: {str(e)}", "classification": "Neutral", "confidence": 0.0, "model": "FinBERT_error"}

    def get_huggingface_sentiment(self, text):
        """
        Get sentiment using HuggingFace transformers - FREE (general purpose, non-financial specific)

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment analysis results
        """
        if self.general_sentiment_pipeline is None:
            return {"error": "HuggingFace general sentiment model not available", "classification": "Neutral", "confidence": 0.0, "model": "None"}

        try:
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

        except Exception as e:
            logger.error(f"HuggingFace general sentiment error for text '{text[:50]}...': {str(e)}")
            return {"error": f"HuggingFace error: {str(e)}", "classification": "Neutral", "confidence": 0.0, "model": "HuggingFace_error"}

    def get_ollama_sentiment(self, text, model="llama2"):
        """
        Get sentiment using legacy Ollama (LangChain wrapper) - FREE

        Args:
            text (str): Text to analyze
            model (str): Ollama model to use (e.g., 'llama2', 'mistral')

        Returns:
            dict: Sentiment analysis results
        """
        if "ollama" not in self.available_models:
            return {"error": "Ollama (LangChain) not available or configured.", "classification": "Neutral", "confidence": 0.0, "model": "None"}
        if not LANGCHAIN_AVAILABLE:
             return {"error": "LangChain not available for Ollama (LangChain) sentiment.", "classification": "Neutral", "confidence": 0.0, "model": "None"}


        try:
            from langchain_community.llms import Ollama

            llm = Ollama(model=model, temperature=0, base_url=self.ollama_base_url)

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
                'confidence': 0.7, # Assign a default confidence as LangChain doesn't usually provide one directly
                'model': f'Ollama-LangChain-{model}',
                'raw_response': response
            }

        except Exception as e:
            logger.error(f"Ollama (LangChain) error for text '{text[:50]}...': {str(e)}")
            return {"error": f"Ollama (LangChain) error: {str(e)}", "classification": "Neutral", "confidence": 0.0, "model": f"Ollama-LangChain-{model}_error"}

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
            logger.warning("TextBlob not available. Install with: pip install textblob")
            return {"error": "TextBlob not available", "classification": "Neutral", "confidence": 0.0, "model": "None"}
        except Exception as e:
            logger.error(f"TextBlob error for text '{text[:50]}...': {str(e)}")
            return {"error": f"TextBlob error: {str(e)}", "classification": "Neutral", "confidence": 0.0, "model": "TextBlob_error"}

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

        # Get Llama sentiment if available (PREFERRED for financial analysis, direct API)
        if "llama" in self.available_models:
            llama_result = self.get_llama_sentiment(text)
            if 'error' not in llama_result:
                results['llama'] = llama_result
            else:
                logger.warning(f"Llama sentiment failed: {llama_result.get('error')}")

        # Get FinBERT sentiment if available (FREE)
        if "finbert" in self.available_models:
            finbert_result = self.get_finbert_sentiment(text)
            if 'error' not in finbert_result:
                results['finbert'] = finbert_result
            else:
                logger.warning(f"FinBERT sentiment failed: {finbert_result.get('error')}")

        # Get HuggingFace general sentiment if available (FREE) - only if finbert not used or failed
        if "huggingface" in self.available_models and 'finbert' not in results:
            hf_result = self.get_huggingface_sentiment(text)
            if 'error' not in hf_result:
                results['huggingface'] = hf_result
            else:
                logger.warning(f"HuggingFace general sentiment failed: {hf_result.get('error')}")


        # Get Ollama sentiment if available (LangChain wrapper) - only if Llama direct API not used or failed
        if "ollama" in self.available_models and 'llama' not in results:
            ollama_lc_result = self.get_ollama_sentiment(text, model=self.llama_model) # Use configured llama model if available
            if 'error' not in ollama_lc_result:
                results['ollama_langchain'] = ollama_lc_result
            else:
                logger.warning(f"Ollama (LangChain) sentiment failed: {ollama_lc_result.get('error')}")


        # Try TextBlob as additional free option
        textblob_result = self.get_textblob_sentiment(text)
        if 'error' not in textblob_result:
            results['textblob'] = textblob_result
        else:
            logger.warning(f"TextBlob sentiment failed: {textblob_result.get('error')}")

        # Calculate consensus with higher weight for Llama and FinBERT
        classifications = []
        confidences = []
        weights = []

        for model, result in results.items():
            if 'classification' in result and result.get('confidence') is not None:
                classifications.append(result['classification'])

                # Get confidence if available
                confidence = result.get('confidence', 0.5) # Default to 0.5 if not explicitly set

                confidences.append(confidence)

                # Assign weights - Llama and FinBERT get higher weights
                if model == 'llama':
                    weights.append(1.5)  # Highest weight for Llama (direct API)
                elif model == 'finbert':
                    weights.append(1.3)  # High weight for FinBERT
                elif model == 'huggingface':
                    weights.append(1.0)
                elif model == 'ollama_langchain': # LangChain Ollama
                    weights.append(0.9)
                else: # VADER, TextBlob
                    weights.append(0.8)

        # Weighted majority vote
        if classifications:
            from collections import defaultdict
            weighted_votes = defaultdict(float)

            for classification, confidence, weight in zip(classifications, confidences, weights):
                weighted_votes[classification] += confidence * weight

            # Get the classification with highest weighted vote
            consensus = max(weighted_votes, key=weighted_votes.get)
            total_weight = sum(weighted_votes.values())
            consensus_confidence = weighted_votes[consensus] / total_weight if total_weight > 0 else 0.0
        else:
            consensus = "Neutral"
            consensus_confidence = 0.0

        return {
            'consensus': consensus,
            'confidence': consensus_confidence,
            'individual_results': results,
            'models_used': len([r for r in results.values() if 'classification' in r]),
            'available_models': self.available_models,
            'llama_direct_api_available': 'llama' in self.available_models
        }

    def get_news_sentiment_for_symbol(self, symbol, days_back=7):
        """
        Get news sentiment for a specific stock symbol using the best available method.
        Filters news to include only articles within the specified days_back window.

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            days_back (int): Number of days to look back for news.
                            Only news published within this period will be analyzed.

        Returns:
            dict: Aggregated sentiment analysis of recent news
        """
        try:
            # Get company name from yfinance
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)

            news_data = ticker.news

            if not news_data:
                logger.info(f"No news data available for {symbol}.")
                return {"symbol": symbol, "company_name": company_name, "overall_sentiment": "Neutral",
                        "sentiment_score": 0.0, "positive_count": 0, "negative_count": 0,
                        "neutral_count": 0, "total_articles": 0, "news_items": [],
                        "analysis_method": "None", "error": "No news data available"}

            # Filter news by days_back
            cutoff_time = int((time.time() - days_back * 24 * 60 * 60)) # Unix timestamp for days_back ago
            recent_news = [item for item in news_data if item.get('providerPublishTime', 0) >= cutoff_time]

            if not recent_news:
                logger.info(f"No recent news found for {symbol} within the last {days_back} days.")
                return {"symbol": symbol, "company_name": company_name, "overall_sentiment": "Neutral",
                        "sentiment_score": 0.0, "positive_count": 0, "negative_count": 0,
                        "neutral_count": 0, "total_articles": 0, "news_items": [],
                        "analysis_method": "None", "error": f"No news found within {days_back} days"}

            # Analyze sentiment for each news item using the best available method
            sentiments = []
            news_items = []

            # Limit to a reasonable number of recent articles to avoid excessive processing time
            for item in recent_news[:10]: # Process up to 10 most recent articles that pass date filter
                title = item.get('title', '')
                if title:
                    sentiment_result = None
                    analysis_method_used = "consensus"

                    # Use Llama direct API if available (highest priority)
                    if "llama" in self.available_models:
                        llama_res = self.get_llama_sentiment(title)
                        if 'error' not in llama_res:
                            sentiment_result = {'consensus': llama_res['classification'], **llama_res}
                            analysis_method_used = "llama"
                        else:
                            logger.warning(f"Llama failed for news item '{title[:50]}...'. Falling back to consensus.")
                    
                    # If Llama wasn't used or failed, try consensus
                    if sentiment_result is None:
                        sentiment_result = self.get_consensus_sentiment(title)
                        analysis_method_used = sentiment_result.get('analysis_method', 'consensus') # Update if consensus picked a specific model

                    if 'error' not in sentiment_result:
                        sentiments.append(sentiment_result)
                        news_items.append({
                            'title': title,
                            'sentiment': sentiment_result.get('consensus', sentiment_result.get('classification', 'Neutral')),
                            'confidence': sentiment_result.get('confidence', 0.0),
                            'model_used': analysis_method_used,
                            'url': item.get('link', ''),
                            'published_timestamp': item.get('providerPublishTime', 0)
                        })
                    else:
                        logger.warning(f"Failed to get sentiment for news item '{title[:50]}...': {sentiment_result.get('error')}")


            # Calculate overall sentiment
            if sentiments:
                positive_count = sum(1 for s in sentiments
                                       if s.get('consensus', s.get('classification', 'Neutral')) == 'Positive')
                negative_count = sum(1 for s in sentiments
                                       if s.get('consensus', s.get('classification', 'Neutral')) == 'Negative')
                neutral_count = sum(1 for s in sentiments
                                     if s.get('consensus', s.get('classification', 'Neutral')) == 'Neutral')

                total = len(sentiments)
                # Weighted average confidence for score
                weighted_score_sum = sum(
                    (1 if s.get('consensus', s.get('classification', 'Neutral')) == 'Positive' else
                     (-1 if s.get('consensus', s.get('classification', 'Neutral')) == 'Negative' else 0)) *
                    s.get('confidence', 0.5) for s in sentiments
                )
                sentiment_score = weighted_score_sum / total if total > 0 else 0.0

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
                logger.info(f"No sentiments could be calculated for {symbol}'s recent news.")

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
            logger.error(f"Error analyzing news sentiment for {symbol}: {str(e)}")
            return {"symbol": symbol, "company_name": "Unknown", "error": f"Error analyzing news sentiment: {str(e)}",
                    "overall_sentiment": "Neutral", "sentiment_score": 0.0, "positive_count": 0, "negative_count": 0,
                    "neutral_count": 0, "total_articles": 0, "news_items": [], "analysis_method": "None"}

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
            batch_results = self._batch_analyze_llama(texts)
            if batch_results:
                return batch_results
            else:
                logger.warning("Batch Llama analysis failed or returned empty, falling back to individual analysis.")

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
                result = self.get_ollama_sentiment(text) # Uses LangChain Ollama
            elif method == 'textblob':
                result = self.get_textblob_sentiment(text)
            elif method == 'consensus':
                result = self.get_consensus_sentiment(text)
            else:
                result = {"error": f"Unknown method: {method}. Available: {self.available_models}",
                          "classification": "Neutral", "confidence": 0.0, "model": "Unknown"}

            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': result
            })

        return results

    def _batch_analyze_llama(self, texts: List[str]) -> List[Dict]:
        """Optimized batch analysis using Llama for multiple texts (direct API)"""
        if "llama" not in self.available_models or not texts:
            logger.warning("Llama batch analysis not possible: Llama model not available or no texts provided.")
            return []

        # Ensure we don't send too many texts in a single batch to avoid excessively long prompts/responses
        if len(texts) > 5:
            logger.warning("Llama batch analysis optimized for up to 5 texts. Processing first 5.")
            texts = texts[:5]

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
                    'num_predict': len(texts) * 5 # Allow enough tokens for all responses + numbering
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
                    sentiment_classification = sentiments[i] if i < len(sentiments) else 'Neutral'
                    confidence = 0.8 if sentiment_classification != 'Neutral' else 0.6 # Basic confidence

                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': {
                            'classification': sentiment_classification,
                            'confidence': confidence,
                            'model': f"{self.llama_model}_batch"
                        }
                    })

                return results

            else:
                logger.error(f"Batch Llama API request failed with status: {response.status_code}, response: {response.text}")
                return []

        except requests.exceptions.Timeout:
            logger.warning(f"Batch Llama analysis timed out for {len(texts)} texts.")
            return []
        except requests.exceptions.ConnectionError:
            logger.error(f"Batch Llama analysis failed due to connection error. Is Ollama server running?")
            return []
        except Exception as e:
            logger.error(f"Batch Llama analysis failed unexpectedly: {e}")
            return []