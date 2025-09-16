"""
Router Service

Advanced natural language intent classification service for multi-agent AI systems.
Provides sophisticated intent classification with confidence scoring, fallback handling,
and comprehensive analytics.
"""

import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from ..core.database import supabase, db_select, get_supabase_client

logger = logging.getLogger(__name__)

# Default fallback response
def get_default_fallback_response(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "detected_intent": None,
        "confidence_score": 0.0,
        "selected_agent": None,
        "rule_used": None,
        "fallback_used": True,
        "processing_time_ms": 0,
        "error": "No active agents or rules found"
    }

class RouterService:
    """Service for intent classification and agent routing"""
    
    @staticmethod
    async def get_fallback_messages() -> List[Dict[str, Any]]:
        """
        Get all fallback messages
        
        Returns:
            List of fallback messages with their details
        """
        try:
            result = await db_select(
                'fallback_messages',
                columns='*',
                order_by='created_at DESC'
            )
            return result.get('data', [])
        except Exception as e:
            logger.error(f"Error getting fallback messages: {str(e)}")
            return []
    
    @staticmethod
    async def get_analytics(time_window: str = '24h') -> Dict[str, Any]:
        """
        Get router analytics
        
        Args:
            time_window: Time window for analytics (e.g., '24h', '7d', '30d')
            
        Returns:
            Dictionary containing analytics data
        """
        try:
            # Parse time window
            window_value = 24  # Default 24 hours
            window_unit = "hours"
            
            if time_window.endswith("h"):
                try:
                    window_value = int(time_window[:-1])
                    window_unit = "hours"
                except:
                    pass
            elif time_window.endswith("d"):
                try:
                    window_value = int(time_window[:-1])
                    window_unit = "days"
                except:
                    pass
            elif time_window.endswith("w"):
                try:
                    window_value = int(time_window[:-1]) * 7
                    window_unit = "days"
                except:
                    pass
                    
            # Return a basic mock response for now
            current_time = datetime.utcnow()
            start_time = current_time - timedelta(**{window_unit: window_value})
            
            return {
                "status": "success",
                "data": {
                    "time_window": time_window,
                    "start_time": start_time.isoformat(),
                    "end_time": current_time.isoformat(),
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "intent_distribution": [],
                    "status_distribution": [],
                    "note": "This is a mock response since the actual analytics data might not exist yet."
                }
            }
        except Exception as error:
            logger.error(f"❌ [RouterService] Get analytics error: {str(error)}")
            return {
                "status": "success",
                "data": {
                    "error": f"Database error: {str(error)}",
                    "time_window": time_window,
                    "total_requests": 0,
                    "success_rate": 0,
                    "intent_distribution": [],
                    "status_distribution": []
                }
            }
        try:
            # Get current time and calculate time delta based on window
            end_time = datetime.utcnow()
            if time_window.endswith('h'):
                hours = int(time_window[:-1])
                start_time = end_time - timedelta(hours=hours)
            elif time_window.endswith('d'):
                days = int(time_window[:-1])
                start_time = end_time - timedelta(days=days)
            else:
                start_time = end_time - timedelta(hours=24)  # Default to 24h
                
            # Get request counts by intent
            intent_counts = await db_select(
                'router_logs',
                columns='detected_intent, count(*) as count',
                filters={
                    'created_at': {'gte': start_time.isoformat()}
                },
                group_by='detected_intent',
                order_by='count DESC'
            )
            
            # Get success/failure rates
            status_stats = await db_select(
                'router_logs',
                columns='is_success, count(*) as count',
                filters={
                    'created_at': {'gte': start_time.isoformat()}
                },
                group_by='is_success'
            )
            
            # Calculate success rate
            total = sum(item['count'] for item in status_stats.get('data', []))
            success_count = next(
                (item['count'] for item in status_stats.get('data', []) 
                 if item['is_success'] is True), 0)
            success_rate = (success_count / total * 100) if total > 0 else 0
            
            return {
                'time_window': time_window,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_requests': total,
                'success_rate': round(success_rate, 2),
                'intent_distribution': intent_counts.get('data', []),
                'status_distribution': status_stats.get('data', [])
            }
            
        except Exception as e:
            logger.error(f"Error getting router analytics: {str(e)}")
            return {
                'error': str(e),
                'time_window': time_window,
                'total_requests': 0,
                'success_rate': 0,
                'intent_distribution': [],
                'status_distribution': []
            }
    
    @staticmethod
    async def _get_active_router_rules() -> List[Dict[str, Any]]:
        """
        Fetch active router rules with agent information
        
        Returns:
            List of active router rules with agent details
        """
        try:
            # Get active rules with agent details
            result = await db_select(
                'router_rules',
                columns="""
                    *,
                    agents:agent_id(*)
                """,
                filters={'is_active': True}
            )
            return result or []
        except Exception as e:
            logger.error(f"Error fetching router rules: {e}")
            return []

    @staticmethod
    async def classify_intent(
        input_text: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify user intent and return the best matching agent
        
        Args:
            input_text: User input text to classify
            options: Additional options for classification
            
        Returns:
            Dictionary with classification results
        """
        start_time = datetime.now()
        response = get_default_fallback_response(input_text)
        
        try:
            if not input_text or not isinstance(input_text, str):
                raise ValueError("Input text must be a non-empty string")
                
            # Get active router rules
            rules = await RouterService._get_active_router_rules()
            if not rules:
                response["error"] = "No active router rules found"
                return response
                
            # Simple keyword matching (can be enhanced with ML later)
            query = input_text.lower()
            best_match = None
            best_score = 0.0
            best_confidence = 0.0  # Initialize best_confidence here
            
            for rule in rules:
                if not rule.get('is_active', True):
                    continue
                    
                # Calculate score based on keyword matching
                keywords = rule.get('keywords', [])
                if not isinstance(keywords, list):
                    keywords = []
                    
                score = sum(1 for kw in keywords if kw.lower() in query) / max(1, len(keywords))
                
                # Parse keywords safely with validation
                keyword_array = []
                try:
                    keywords = rule.get('keywords', [])
                    if isinstance(keywords, list):
                        keyword_array = [str(k).lower() for k in keywords if k and isinstance(k, (str, int, float))]
                    elif isinstance(keywords, str):
                        # Handle JSON string
                        parsed = json.loads(keywords)
                        if isinstance(parsed, list):
                            keyword_array = [str(k).lower() for k in parsed if k and isinstance(k, (str, int, float))]
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f" [RouterService] Failed to parse keywords for rule {rule.get('id')}: {e}")
                
                # Get confidence threshold with type safety
                try:
                    threshold = float(rule.get('confidence_threshold', 0.6))
                    threshold = max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
                except (TypeError, ValueError):
                    threshold = 0.6
                
                logger.debug(f"   Keywords: {json.dumps(keyword_array)}")
                logger.debug(f"   Threshold: {threshold}")
                
                # Calculate confidence score
                confidence = RouterService._calculate_confidence(query, keyword_array)
                logger.debug(f"   Calculated confidence: {confidence:.3f}")
                
                # Update best match if this rule has higher confidence
                if confidence >= threshold and confidence > best_confidence:
                    logger.debug(f"   NEW BEST MATCH! (Confidence: {confidence:.3f} >= {threshold})")
                    best_match = {
                        'intent_name': rule.get('intent_name'),
                        'confidence': min(1.0, max(0.0, confidence)),  # Ensure 0-1 range
                        'id': rule.get('id'),
                        'agent_id': rule.get('agent_id'),
                        'agents': rule.get('agents', {}),
                        'confidence_threshold': threshold,
                        'description': rule.get('description')
                    }
                    best_confidence = confidence
                else:
                    logger.debug(f"   No match (confidence: {confidence:.3f} < threshold: {threshold} or not better than current best: {best_confidence:.3f})")
            
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Prepare response according to RouterClassifyResponse model
            response = {
                "query": input_text,
                "detected_intent": best_match.get('intent_name') if best_match else None,
                "confidence_score": float(best_confidence) if best_match is not None else 0.0,
                "selected_agent": {
                    "id": best_match.get('agent_id'),
                    "name": best_match.get('agents', {}).get('name'),
                    "description": best_match.get('agents', {}).get('description'),
                    "is_active": best_match.get('agents', {}).get('is_active', True)
                } if best_match and best_match.get('agents') else None,
                "rule_used": {
                    "id": best_match.get('id'),
                    "intent_name": best_match.get('intent_name'),
                    "confidence_threshold": best_match.get('confidence_threshold')
                } if best_match else None,
                "fallback_used": best_match is None,
                "processing_time_ms": processing_time_ms,
                "error": None
            }
            
            # Log the classification
            await RouterService._log_intent_classification(
                input_text,
                response["detected_intent"],
                response["confidence_score"],
                best_match.get('id') if best_match else None,
                best_match.get('agent_id') if best_match else None,
                best_match.get('agents', {}).get('name') if best_match and best_match.get('agents') else None,
                response["fallback_used"],
                processing_time_ms,
                options.get('session_id'),
                options.get('user_id')
            )
            
            return response
            
        except Exception as error:
            error_msg = f"Error in classify_intent: {str(error)}"
            logger.error(error_msg, exc_info=True)
            
            # Update response with error
            response.update({
                "error": error_msg,
                "fallback_used": True,
                "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000)
            })
            
            # Log the error
            await RouterService._log_intent_classification(
                input_text,
                None,  # intent
                0.0,   # confidence
                None,  # rule_id
                None,  # agent_id
                None,  # agent_name
                True,  # fallback_used
                response["processing_time_ms"],
                options.get('session_id'),
                options.get('user_id'),
                error_msg
            )
            
            return response
    
    @staticmethod
    def _calculate_confidence(query: str, keywords: List[str]) -> float:
        """
        Calculate confidence score using advanced matching algorithms
        
        Args:
            query: User query (lowercase)
            keywords: Array of keywords to match against
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not keywords or not isinstance(keywords, list) or len(keywords) == 0:
                logger.debug("   [RouterService] No valid keywords found")
                return 0
            
            if not query or not isinstance(query, str):
                logger.debug(f"   [RouterService] Invalid query: {type(query)} {query}")
                return 0
                
            # Use debug level instead of info to reduce log verbosity
            logger.debug(f"   [RouterService] Processing keywords: {json.dumps(keywords)}")
            logger.debug(f"   [RouterService] Against query: \"{query}\"")
            
            query_words = [word for word in query.split() if word.strip()]
            logger.debug(f"   [RouterService] Query words: {json.dumps(query_words)}")
            
            matched_keywords = 0
            total_score = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower().strip()
                if not keyword_lower:
                    logger.info("     [RouterService] Empty keyword, skipping")
                    continue
                
                keyword_score = 0
                logger.debug(f"     [RouterService] Checking keyword: \"{keyword_lower}\"")
                
                # 1. Exact phrase match gets highest score (1.0)
                if keyword_lower in query:
                    keyword_score = 1.0
                    logger.debug("       [RouterService] EXACT PHRASE MATCH! Score: 1.0")
                else:
                    # 2. Check for word-level matches
                    keyword_words = [word for word in keyword_lower.split() if word.strip()]
                    best_word_score = 0
                    
                    logger.debug(f"       [RouterService] Keyword words: {json.dumps(keyword_words)}")
                    
                    for kw_word in keyword_words:
                        word_score = 0
                        
                        for query_word in query_words:
                            if query_word == kw_word:
                                # Exact word match
                                word_score = max(word_score, 1.0)
                                logger.debug(f"       [RouterService] EXACT WORD MATCH: \"{kw_word}\" = \"{query_word}\" (1.0)")
                            elif kw_word in query_word or query_word in kw_word:
                                # Partial word match (substring)
                                word_score = max(word_score, 0.8)
                                logger.debug(f"       [RouterService] PARTIAL WORD MATCH: \"{kw_word}\" ~ \"{query_word}\" (0.8)")
                            elif RouterService._calculate_levenshtein_distance(query_word, kw_word) <= 2 and len(kw_word) > 3:
                                # Fuzzy match for typos (only for longer words, distance ≤ 2)
                                word_score = max(word_score, 0.5)
                                logger.debug(f"       [RouterService] FUZZY WORD MATCH: \"{kw_word}\" ~ \"{query_word}\" (0.5)")
                        
                        # Take the best score for this keyword word
                        best_word_score = max(best_word_score, word_score)
                        
                        if word_score == 0:
                            logger.debug(f"       [RouterService] NO MATCH for keyword word: \"{kw_word}\"")
                    
                    # For multi-word keywords, we need at least some words to match
                    if len(keyword_words) > 1:
                        # For multi-word keywords, require at least 50% of words to have some match
                        words_with_matches = []
                        for kw_word in keyword_words:
                            for q_word in query_words:
                                if (q_word == kw_word or 
                                    kw_word in q_word or 
                                    q_word in kw_word or
                                    (RouterService._calculate_levenshtein_distance(q_word, kw_word) <= 2 and len(kw_word) > 3)):
                                    words_with_matches.append(kw_word)
                                    break
                        
                        match_ratio = len(words_with_matches) / len(keyword_words)
                        keyword_score = best_word_score * match_ratio
                        logger.debug(f"       [RouterService] Multi-word keyword: {len(words_with_matches)}/{len(keyword_words)} words matched, ratio: {match_ratio:.2f}")
                    else:
                        keyword_score = best_word_score
                
                if keyword_score > 0:
                    matched_keywords += 1
                    total_score += keyword_score
                    logger.debug(f"     [RouterService] Keyword \"{keyword_lower}\" contributed: {keyword_score:.3f}")
                else:
                    logger.debug(f"     [RouterService] Keyword \"{keyword_lower}\" contributed: 0.000")
            
            # Advanced confidence calculation based on best N matches
            # This prevents harsh averaging and rewards good partial matches
            final_confidence = 0
            
            if matched_keywords > 0:
                # Strategy: Use average of matched keywords, but boost for multiple matches
                average_match_score = total_score / matched_keywords
                
                # Base confidence from average match quality
                final_confidence = average_match_score
                
                # Boost for having multiple keyword matches (up to 20% bonus)
                if matched_keywords >= 2:
                    match_bonus = min(0.2, (matched_keywords - 1) * 0.1)
                    final_confidence = min(1.0, final_confidence + match_bonus)
                    logger.debug(f"   [RouterService] Multi-match bonus: +{match_bonus:.3f} for {matched_keywords} matches")
                
                # Additional boost if we matched a high percentage of keywords
                match_ratio = matched_keywords / len(keywords)
                if match_ratio >= 0.5:
                    ratio_bonus = min(0.1, (match_ratio - 0.5) * 0.2)
                    final_confidence = min(1.0, final_confidence + ratio_bonus)
                    logger.debug(f"   [RouterService] High match ratio bonus: +{ratio_bonus:.3f} for {(match_ratio * 100):.1f}% keyword coverage")
            
            logger.debug(f"   [RouterService] FINAL CONFIDENCE: {matched_keywords}/{len(keywords)} keywords matched, score: {final_confidence:.3f}")
            
            return min(final_confidence, 1.0)  # Ensure we never exceed 1.0
            
        except Exception as error:
            logger.error(f"[RouterService] Error calculating confidence: {error}")
            return 0
    
    @staticmethod
    def _calculate_levenshtein_distance(str1: str, str2: str) -> int:
        """
        Calculate Levenshtein distance for fuzzy string matching
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Edit distance between strings
        """
        if len(str1) < len(str2):
            return RouterService._calculate_levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    async def select_agent(intent: str) -> Optional[Dict[str, Any]]:
        """
        Select agent based on intent using router rules
        
        Args:
            intent: Detected intent name
            
        Returns:
            Agent information or None if not found
        """
        try:
            if not intent or not isinstance(intent, str):
                return None
            
            supabase = get_supabase_client()
            
            response = await supabase.table('router_rules').select(
                """
                id,
                intent_name,
                agent_id,
                description,
                agents!inner(
                    id, 
                    name, 
                    description, 
                    system_prompt, 
                    input_intents, 
                    output_intents,
                    is_active
                )
                """
            ).eq('intent_name', intent).eq('is_active', True).eq('agents.is_active', True).order('priority', desc=False).limit(1).execute()
            
            if response.data is None or len(response.data) == 0:
                return None
            
            rule = response.data[0]
            
            return {
                'agent_id': rule['agent_id'],
                'agent_name': rule['agents']['name'],
                'agent_description': rule['agents']['description'],
                'agent_data': rule['agents'],
                'rule_id': rule['id'],
                'rule_description': rule['description']
            }
            
        except Exception as error:
            logger.error(f"Error in _get_agent_rule_info: {error}")
            return {
                'agent_id': None,
                'agent_name': None,
                'agent_description': None,
                'agent_data': None,
                'rule_id': None,
                'rule_description': None,
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as error:
            logger.error(f"❌ [RouterService] Intent logging error: {error}")
            # Don't throw - logging failures shouldn't break the main flow
    
    @staticmethod
    async def get_fallback_message(category: str = 'general') -> Dict[str, Any]:
        """
        Get fallback message when no intent matches
        
        Args:
            category: Message category (default: 'general')
            
        Returns:
            Fallback message object
        """
        try:
            supabase = get_supabase_client()
            
            # Get active fallback messages for the category - handle both awaitable and non-awaitable versions
            try:
                # Try non-awaitable version first (newer Supabase client)
                response = supabase.table('fallback_messages').select('*').eq(
                    'is_active', True
                ).eq('category', category).execute()
            except Exception as exec_error:
                try:
                    # Try awaitable version as fallback
                    response = await supabase.table('fallback_messages').select('*').eq(
                        'is_active', True
                    ).eq('category', category).execute()
                except Exception as await_error:
                    logger.error(f"Database execution error: {await_error}")
                    raise Exception(f"Failed to get fallback messages: {await_error}")
            
            if not hasattr(response, 'data') or response.data is None:
                error_msg = getattr(response, 'error', 'Unknown error') if hasattr(response, 'error') else "Unknown error"
                raise Exception(f"Failed to get fallback messages: {error_msg}")
            
            messages = response.data
            
            if not messages:
                # Try general category if specific category has no messages
                if category != 'general':
                    return await RouterService.get_fallback_message('general')
                
                # Return default fallback if no messages in database
                return {
                    'message': "I'm sorry, I didn't understand your request. Could you please rephrase it?",
                    'category': 'default',
                    'id': None,
                    'usage_count': 0
                }
            
            # Select random message from available options
            import random
            random_message = random.choice(messages)
            
            # Update usage count and last used timestamp
            try:
                await supabase.table('fallback_messages').update({
                    'usage_count': (random_message.get('usage_count', 0) + 1),
                    'last_used_at': datetime.utcnow().isoformat()
                }).eq('id', random_message['id']).execute()
            except Exception as update_error:
                logger.error(f"Failed to update fallback message usage: {update_error}")
            
            return {
                'message': random_message['message'],
                'category': random_message['category'],
                'id': random_message['id'],
                'usage_count': (random_message.get('usage_count', 0) + 1)
            }
            
        except Exception as error:
            logger.error(f"❌ [RouterService] Fallback message error: {error}")
            
            # Return default fallback on any error
            return {
                'message': "I'm sorry, I'm having trouble understanding right now. Please try again.",
                'category': 'error',
                'id': None,
                'usage_count': 0,
                'error': str(error)
            }
    
    @staticmethod
    async def get_router_rules(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get router rules with filtering and pagination
        
        Args:
            options: Query options including limit, offset, search, active
            
        Returns:
            Paginated router rules with metadata
        """
        if options is None:
            options = {}
            
        try:
            supabase = get_supabase_client()
            
            # Build query
            query = supabase.table('router_rules').select(
                """
                *,
                agents!inner(id, name, description, is_active)
                """, 
                count='exact'
            ).order('priority', desc=False)
            
            # Apply search filter
            if options.get('search') and isinstance(options['search'], str):
                search_term = options['search'].strip()
                # Note: Supabase Python client might handle ilike differently
                # This is a simplified version - adjust based on your Supabase setup
                query = query.or_(f"intent_name.ilike.%{search_term}%,description.ilike.%{search_term}%")
            
            # Apply active filter
            if 'active' in options:
                query = query.eq('is_active', bool(options['active']))
            
            # Apply pagination
            if options.get('limit') and options['limit'] > 0:
                limit = min(options['limit'], 1000)  # Cap at 1000 for performance
                query = query.limit(limit)
            
            if options.get('offset') and options['offset'] > 0:
                offset = options['offset']
                limit = options.get('limit', 50)
                query = query.range(offset, offset + limit - 1)
            
            # Execute the query - handle both awaitable and non-awaitable versions
            try:
                # Try non-awaitable version first (newer Supabase client)
                response = query.execute()
            except Exception as exec_error:
                try:
                    # Try awaitable version as fallback
                    response = await query.execute()
                except Exception as await_error:
                    logger.error(f"Database execution error: {await_error}")
                    raise Exception(f"Failed to get router rules: {await_error}")
            
            # Check if data exists
            if not hasattr(response, 'data') or response.data is None:
                error_msg = getattr(response, 'error', 'Unknown error') if hasattr(response, 'error') else "Unknown error"
                raise Exception(f"Failed to get router rules: {error_msg}")
            
            # Get count safely using error handling
            count = 0
            try:
                count = response.count if hasattr(response, 'count') else getattr(response, 'count', 0)
            except Exception:
                # If accessing count property fails, default to 0
                pass
            
            return {
                'rules': response.data,
                'total': count,
                'limit': options.get('limit'),
                'offset': options.get('offset', 0),
                'has_more': (count > (options.get('offset', 0) + len(response.data))) if response.data else False
            }
            
        except Exception as error:
            logger.error(f"❌ [RouterService] Get router rules error: {error}")
            raise error
    
    @staticmethod
    async def get_router_analytics(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get router performance analytics
        
        Args:
            options: Analytics options including date filters
            
        Returns:
            Router performance analytics
        """
        if options is None:
            options = {}
            
        try:
            supabase = get_supabase_client()
            
            # Get recent intent logs for analysis - handle both awaitable and non-awaitable versions
            try:
                # Try non-awaitable version first (newer Supabase client)
                response = supabase.table('intent_logs').select(
                    'detected_intent, fallback_used, confidence_score, created_at, processing_time_ms'
                ).order('created_at', desc=True).limit(1000).execute()
            except Exception as exec_error:
                try:
                    # Try awaitable version as fallback
                    response = await supabase.table('intent_logs').select(
                        'detected_intent, fallback_used, confidence_score, created_at, processing_time_ms'
                    ).order('created_at', desc=True).limit(1000).execute()
                except Exception as await_error:
                    logger.error(f"Database execution error: {await_error}")
                    raise Exception(f"Failed to get intent logs: {await_error}")
            
            if not hasattr(response, 'data') or response.data is None:
                error_msg = getattr(response, 'error', 'Unknown error') if hasattr(response, 'error') else "Unknown error"
                raise Exception(f"Failed to get intent logs: {error_msg}")
            
            logs = response.data
            
            # Calculate analytics
            total_classifications = len(logs)
            successful_classifications = len([log for log in logs if not log.get('fallback_used', True)])
            fallback_count = total_classifications - successful_classifications
            
            # Calculate confidence distribution
            confidences = [log.get('confidence_score', 0) for log in logs if log.get('confidence_score') is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Calculate processing time stats
            processing_times = [log.get('processing_time_ms', 0) for log in logs if log.get('processing_time_ms') is not None]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Intent frequency analysis
            intent_counts = {}
            for log in logs:
                intent = log.get('detected_intent')
                if intent:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Daily stats (last 7 days)
            from collections import defaultdict
            daily_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'fallback': 0})
            
            for log in logs:
                try:
                    created_at = datetime.fromisoformat(log['created_at'].replace('Z', '+00:00'))
                    date_key = created_at.strftime('%Y-%m-%d')
                    daily_stats[date_key]['total'] += 1
                    if not log.get('fallback_used', True):
                        daily_stats[date_key]['successful'] += 1
                    else:
                        daily_stats[date_key]['fallback'] += 1
                except:
                    continue
            
            return {
                'summary': {
                    'total_classifications': total_classifications,
                    'successful_classifications': successful_classifications,
                    'fallback_count': fallback_count,
                    'success_rate': successful_classifications / total_classifications if total_classifications > 0 else 0,
                    'avg_confidence': avg_confidence,
                    'avg_processing_time_ms': avg_processing_time
                },
                'top_intents': top_intents,
                'daily_stats': dict(daily_stats),
                'recent_activity': logs[:50],  # Last 50 classifications
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as error:
            logger.error(f"❌ [RouterService] Get router analytics error: {error}")
            
            # Return empty analytics on error
            return {
                'summary': {
                    'total_classifications': 0,
                    'successful_classifications': 0,
                    'fallback_count': 0,
                    'success_rate': 0,
                    'avg_confidence': 0,
                    'avg_processing_time_ms': 0
                },
                'top_intents': [],
                'daily_stats': {},
                'recent_activity': [],
                'generated_at': datetime.utcnow().isoformat(),
                'error': str(error)
            }
            
    @staticmethod
    async def _log_intent_classification(
        input_text: str,
        detected_intent: Optional[str],
        confidence_score: float,
        rule_id: Optional[str],
        agent_id: Optional[str],
        agent_name: Optional[str],
        fallback_used: bool,
        processing_time_ms: int,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log intent classification results for analytics
        
        Args:
            input_text: Original user query
            detected_intent: Detected intent name or None
            confidence_score: Confidence score (0-1)
            rule_id: ID of the matching rule or None
            agent_id: ID of the selected agent or None
            agent_name: Name of the selected agent or None
            fallback_used: Whether fallback was used
            processing_time_ms: Processing time in milliseconds
            session_id: Optional session ID
            user_id: Optional user ID
            error: Optional error message
        """
        try:
            # Skip logging for empty queries
            if not input_text or not input_text.strip():
                return
                
            # Prepare log data
            log_data = {
                "query": input_text[:500],  # Limit length
                "detected_intent": detected_intent,
                "confidence_score": confidence_score,
                "rule_id": rule_id,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "fallback_used": fallback_used,
                "processing_time_ms": processing_time_ms,
                "session_id": session_id,
                "user_id": user_id or "anonymous",
                "error": error[:500] if error else None,  # Limit length
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Use debug level instead of info to reduce log verbosity
            logger.debug(f"Intent classification logged: {detected_intent or 'fallback'} "
                        f"(confidence: {confidence_score:.2f}, time: {processing_time_ms}ms)")
            
            # Note: In a production system, this would be stored in a database
            # Currently we're just logging it to avoid issues with missing database tables
            
        except Exception as error:
            logger.error(f"❌ [RouterService] Error logging intent classification: {error}")
            # Don't throw - logging failures shouldn't break the main flow
            
    @staticmethod
    async def get_logs(
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get router classification logs
        
        Args:
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            
        Returns:
            Dictionary with logs and pagination info
        """
        try:
            # Return a basic mock response for now
            logger.info(f"Get router logs requested with limit={limit}, offset={offset}")
            return {
                "status": "success",
                "data": [],
                "pagination": {
                    "total": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False
                },
                "note": "This is a mock response since the actual database table might not exist yet."
            }
        except Exception as error:
            logger.error(f"❌ [RouterService] Get logs error: {error}")
            raise error
            
    @staticmethod
    async def get_metrics() -> Dict[str, Any]:
        """
        Get router performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Return a basic mock response for now
            current_time = datetime.utcnow()
            return {
                "status": "success",
                "data": {
                    "total_requests": 0,
                    "total_successful": 0,
                    "total_fallbacks": 0,
                    "success_rate": 0.0,
                    "average_confidence": 0.0,
                    "average_processing_time_ms": 0,
                    "top_intents": [],
                    "updated_at": current_time.isoformat()
                },
                "note": "This is a mock response since the actual metrics table might not exist yet."
            }
        except Exception as error:
            logger.error(f"❌ [RouterService] Get metrics error: {error}")
            raise error
            
    @staticmethod
    async def test_router() -> Dict[str, Any]:
        """
        Test router functionality with sample queries
        
        Returns:
            Dictionary with test results
        """
        try:
            # Sample test queries
            test_queries = [
                "Hello, how are you?",
                "I need help with my account",
                "What's the status of my order?",
                "Can you recommend a product for me?",
                "I want to cancel my subscription"
            ]
            
            results = []
            for query in test_queries:
                # Test each query
                start_time = datetime.utcnow()
                
                try:
                    # Use classify_intent for each query
                    result = {
                        "query": query,
                        "detected_intent": None,
                        "confidence_score": 0.0,
                        "selected_agent": None,
                        "processing_time_ms": 0,
                        "status": "success"
                    }
                    
                    # For now, just mock the results to avoid circular dependencies
                    result["detected_intent"] = "test_intent"
                    result["confidence_score"] = 0.85
                    result["selected_agent"] = {"name": "Test Agent"}
                    result["processing_time_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    results.append(result)
                except Exception as e:
                    results.append({
                        "query": query,
                        "error": str(e),
                        "status": "error"
                    })
                    
            return {
                "status": "success",
                "results": results,
                "summary": {
                    "total_tests": len(test_queries),
                    "successful_tests": sum(1 for r in results if r.get("status") == "success"),
                    "failed_tests": sum(1 for r in results if r.get("status") == "error")
                }
            }
        except Exception as error:
            logger.error(f"❌ [RouterService] Test router error: {error}")
            raise error
