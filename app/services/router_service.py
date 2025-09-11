"""
Router Service

Advanced natural language intent classification service for multi-agent AI systems.
Provides sophisticated intent classification with confidence scoring, fallback handling,
and comprehensive analytics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from ..core.database import get_supabase_client
from ..core.models import RouterRuleModel, IntentLogModel, FallbackMessageModel

logger = logging.getLogger(__name__)


class RouterService:
    """Service for intent classification and agent routing"""
    
    @staticmethod
    async def classify_intent(input_text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify user intent using advanced natural language processing
        
        Args:
            input_text: User query/input text
            options: Classification options including session_id, user_id, min_confidence
            
        Returns:
            Classification result with intent, confidence, and agent info
        """
        start_time = datetime.now()
        
        if options is None:
            options = {}
            
        try:
            if not input_text or not isinstance(input_text, str):
                raise ValueError('Input must be a non-empty string')
            
            logger.info(f"🔍 [RouterService] Classifying intent for query: \"{input_text}\"")
            
            # Fetch active router rules with agent information, ordered by priority
            supabase = get_supabase_client()
            
            response = await supabase.table('router_rules').select(
                """
                id,
                intent_name,
                keywords,
                agent_id,
                priority,
                confidence_threshold,
                description,
                agents!inner(id, name, description, is_active)
                """
            ).eq('is_active', True).eq('agents.is_active', True).order('priority', desc=False).execute()
            
            if response.data is None:
                raise Exception(f"Failed to fetch router rules: {getattr(response, 'error', 'Unknown error')}")
            
            rules = response.data
            logger.info(f"📋 [RouterService] Found {len(rules)} active router rules")
            
            query = input_text.lower().strip()
            best_match = None
            best_confidence = 0
            
            # Evaluate each rule for confidence scoring
            for rule in rules:
                logger.info(f"\n🔎 [RouterService] Evaluating rule: {rule['intent_name']} (Priority: {rule['priority']})")
                
                # Parse keywords safely
                try:
                    if isinstance(rule['keywords'], list):
                        keyword_array = rule['keywords']
                    elif isinstance(rule['keywords'], str):
                        keyword_array = json.loads(rule['keywords'])
                    else:
                        keyword_array = []
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"⚠️  [RouterService] Failed to parse keywords for rule {rule['intent_name']}: {e}")
                    keyword_array = []
                
                logger.info(f"   Keywords: {json.dumps(keyword_array)}")
                logger.info(f"   Threshold: {rule['confidence_threshold'] or 0.6}")
                
                confidence = RouterService._calculate_confidence(query, keyword_array)
                threshold = options.get('min_confidence', rule['confidence_threshold'] or 0.6)
                
                logger.info(f"   Calculated confidence: {confidence:.3f}")
                logger.info(f"   Required threshold: {threshold}")
                
                if confidence >= threshold and confidence > best_confidence:
                    logger.info(f"   ✅ NEW BEST MATCH! (Previous best: {best_confidence:.3f})")
                    best_match = {
                        'intent': rule['intent_name'],
                        'confidence': confidence,
                        'rule_id': rule['id'],
                        'agent_id': rule['agent_id'],
                        'agent_name': rule['agents']['name'],
                        'agent_description': rule['agents']['description'],
                        'rule_description': rule['description']
                    }
                    best_confidence = confidence
                else:
                    logger.info(f"   ❌ No match (confidence: {confidence:.3f} < threshold: {threshold} or not better than current best: {best_confidence:.3f})")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if best_match:
                logger.info(f"\n🎯 [RouterService] FINAL RESULT: Intent \"{best_match['intent']}\" selected with confidence {best_match['confidence']:.3f}")
                logger.info(f"   Agent: {best_match['agent_name']}")
            else:
                logger.info(f"\n❌ [RouterService] NO MATCH FOUND - will use fallback")
            
            # Log the classification attempt
            await RouterService._log_intent(
                input_text,
                best_match['intent'] if best_match else None,
                best_match['agent_id'] if best_match else None,
                best_match['agent_name'] if best_match else None,
                best_match['confidence'] if best_match else 0,
                best_match['rule_id'] if best_match else None,
                not best_match,
                processing_time,
                options.get('session_id'),
                options.get('user_id')
            )
            
            return best_match or {
                'intent': None,
                'confidence': 0,
                'rule_id': None,
                'agent_id': None,
                'agent_name': None,
                'fallback_needed': True,
                'processing_time_ms': processing_time
            }
            
        except Exception as error:
            logger.error(f"❌ [RouterService] Intent classification error: {error}")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log the error attempt
            try:
                await RouterService._log_intent(
                    input_text,
                    None,
                    None,
                    None,
                    0,
                    None,
                    True,
                    processing_time,
                    options.get('session_id'),
                    options.get('user_id')
                )
            except Exception as log_error:
                logger.error(f"Failed to log classification error: {log_error}")
            
            return {
                'intent': None,
                'confidence': 0,
                'rule_id': None,
                'agent_id': None,
                'agent_name': None,
                'fallback_needed': True,
                'error': str(error),
                'processing_time_ms': processing_time
            }
    
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
                logger.info("   ❌ [RouterService] No valid keywords found")
                return 0
            
            if not query or not isinstance(query, str):
                logger.info(f"   ❌ [RouterService] Invalid query: {type(query)} {query}")
                return 0
            
            logger.info(f"   🔍 [RouterService] Processing keywords: {json.dumps(keywords)}")
            logger.info(f"   🔍 [RouterService] Against query: \"{query}\"")
            
            query_words = [word for word in query.split() if word.strip()]
            logger.info(f"   📝 [RouterService] Query words: {json.dumps(query_words)}")
            
            matched_keywords = 0
            total_score = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower().strip()
                if not keyword_lower:
                    logger.info("     ⚠️  [RouterService] Empty keyword, skipping")
                    continue
                
                keyword_score = 0
                logger.info(f"     🔎 [RouterService] Checking keyword: \"{keyword_lower}\"")
                
                # 1. Exact phrase match gets highest score (1.0)
                if keyword_lower in query:
                    keyword_score = 1.0
                    logger.info("       ✅ EXACT PHRASE MATCH! Score: 1.0")
                else:
                    # 2. Check for word-level matches
                    keyword_words = [word for word in keyword_lower.split() if word.strip()]
                    best_word_score = 0
                    
                    logger.info(f"       📝 [RouterService] Keyword words: {json.dumps(keyword_words)}")
                    
                    for kw_word in keyword_words:
                        word_score = 0
                        
                        for query_word in query_words:
                            if query_word == kw_word:
                                # Exact word match
                                word_score = max(word_score, 1.0)
                                logger.info(f"       ✅ EXACT WORD MATCH: \"{kw_word}\" = \"{query_word}\" (1.0)")
                            elif kw_word in query_word or query_word in kw_word:
                                # Partial word match (substring)
                                word_score = max(word_score, 0.8)
                                logger.info(f"       ✅ PARTIAL WORD MATCH: \"{kw_word}\" ~ \"{query_word}\" (0.8)")
                            elif RouterService._calculate_levenshtein_distance(query_word, kw_word) <= 2 and len(kw_word) > 3:
                                # Fuzzy match for typos (only for longer words, distance ≤ 2)
                                word_score = max(word_score, 0.5)
                                logger.info(f"       ✅ FUZZY WORD MATCH: \"{kw_word}\" ~ \"{query_word}\" (0.5)")
                        
                        # Take the best score for this keyword word
                        best_word_score = max(best_word_score, word_score)
                        
                        if word_score == 0:
                            logger.info(f"       ❌ NO MATCH for keyword word: \"{kw_word}\"")
                    
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
                        logger.info(f"       📊 Multi-word keyword: {len(words_with_matches)}/{len(keyword_words)} words matched, ratio: {match_ratio:.2f}")
                    else:
                        keyword_score = best_word_score
                
                if keyword_score > 0:
                    matched_keywords += 1
                    total_score += keyword_score
                    logger.info(f"     📈 Keyword \"{keyword_lower}\" contributed: {keyword_score:.3f}")
                else:
                    logger.info(f"     📉 Keyword \"{keyword_lower}\" contributed: 0.000")
            
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
                    logger.info(f"   🚀 Multi-match bonus: +{match_bonus:.3f} for {matched_keywords} matches")
                
                # Additional boost if we matched a high percentage of keywords
                match_ratio = matched_keywords / len(keywords)
                if match_ratio >= 0.5:
                    ratio_bonus = min(0.1, (match_ratio - 0.5) * 0.2)
                    final_confidence = min(1.0, final_confidence + ratio_bonus)
                    logger.info(f"   🎯 High match ratio bonus: +{ratio_bonus:.3f} for {(match_ratio * 100):.1f}% keyword coverage")
            
            logger.info(f"   🎯 [RouterService] FINAL CONFIDENCE: {matched_keywords}/{len(keywords)} keywords matched, score: {final_confidence:.3f}")
            
            return min(final_confidence, 1.0)  # Ensure we never exceed 1.0
            
        except Exception as error:
            logger.error(f"❌ [RouterService] Error calculating confidence: {error}")
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
            logger.error(f"❌ [RouterService] Agent selection error: {error}")
            return None
    
    @staticmethod
    async def _log_intent(
        query: str,
        intent: Optional[str],
        selected_agent_id: Optional[str],
        selected_agent_name: Optional[str],
        confidence: float,
        rule_id: Optional[str],
        fallback_used: bool,
        processing_time_ms: float,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Log intent classification attempt to database
        
        Args:
            query: Original user query
            intent: Detected intent
            selected_agent_id: Selected agent ID
            selected_agent_name: Selected agent name
            confidence: Confidence score
            rule_id: Matched rule ID
            fallback_used: Whether fallback was used
            processing_time_ms: Processing time in milliseconds
            session_id: Session identifier
            user_id: User identifier
        """
        try:
            log_data = {
                'user_query': query,
                'detected_intent': intent,
                'confidence_score': confidence,
                'selected_agent_id': selected_agent_id,
                'selected_agent_name': selected_agent_name,
                'rule_id': rule_id,
                'fallback_used': fallback_used,
                'processing_time_ms': processing_time_ms,
                'session_id': session_id,
                'user_id': user_id,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Validate using model if available
            try:
                validated_data = IntentLogModel.validate(log_data)
            except:
                validated_data = log_data  # Use raw data if validation fails
            
            supabase = get_supabase_client()
            await supabase.table('intent_logs').insert(validated_data).execute()
            
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
            
            # Get active fallback messages for the category
            response = await supabase.table('fallback_messages').select('*').eq(
                'is_active', True
            ).eq('category', category).execute()
            
            if response.data is None:
                raise Exception(f"Failed to get fallback messages: {getattr(response, 'error', 'Unknown error')}")
            
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
            
            response = await query.execute()
            
            if response.data is None:
                raise Exception(f"Failed to get router rules: {getattr(response, 'error', 'Unknown error')}")
            
            return {
                'rules': response.data,
                'total': getattr(response, 'count', 0),
                'limit': options.get('limit'),
                'offset': options.get('offset', 0),
                'has_more': (getattr(response, 'count', 0) > 
                           (options.get('offset', 0) + len(response.data))) if response.data else False
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
            
            # Get recent intent logs for analysis
            response = await supabase.table('intent_logs').select(
                'detected_intent, fallback_used, confidence_score, created_at, processing_time_ms'
            ).order('created_at', desc=True).limit(1000).execute()
            
            if response.data is None:
                raise Exception(f"Failed to get intent logs: {getattr(response, 'error', 'Unknown error')}")
            
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
