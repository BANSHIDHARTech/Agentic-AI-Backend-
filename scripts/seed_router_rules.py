#!/usr/bin/env python3

"""
Router Rules Seeding Script

Seeds the database with sample router rules and fallback messages
for the Router/Commander Agent system.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import supabase, db_insert

async def seed_router_rules():
    """Seed router rules and fallback messages"""
    print('🌱 Seeding Router Rules and Fallback Messages...')
    
    try:
        # Check if rules already exist
        existing_rules = supabase.from_('router_rules').select('id').limit(1).execute()
        
        if existing_rules.data and len(existing_rules.data) > 0:
            print('✅ Router rules already exist, skipping seed')
            return
        
        # Get agent IDs for rules
        agents_result = supabase.from_('agents').select('id, name').execute()
        agents = agents_result.data or []
        
        sim_swap_agent = next((a for a in agents if 'SIM Swap' in a['name']), None)
        balance_agent = next((a for a in agents if a['name'] == 'PostpaidBalanceAgent'), None)
        auth_agent = next((a for a in agents if a['name'] == 'AuthAgent'), None)
        support_agent = next((a for a in agents if a['name'] == 'SupportAgent'), None)
        
        # Define router rules
        rules = [
            {
                'intent_name': 'sim_swap_request',
                'keywords': ['sim', 'swap', 'change', 'replace', 'sim card'],
                'agent_id': sim_swap_agent['id'] if sim_swap_agent else None,
                'priority': 10,
                'confidence_threshold': 0.8,
                'description': 'Routes SIM swap and SIM card replacement requests'
            },
            {
                'intent_name': 'balance_inquiry',
                'keywords': ['balance', 'account', 'bill', 'payment', 'due', 'amount'],
                'agent_id': balance_agent['id'] if balance_agent else None,
                'priority': 20,
                'confidence_threshold': 0.7,
                'description': 'Routes balance and account inquiry requests'
            },
            {
                'intent_name': 'authentication_required',
                'keywords': ['login', 'authenticate', 'verify', 'password', 'access'],
                'agent_id': auth_agent['id'] if auth_agent else None,
                'priority': 30,
                'confidence_threshold': 0.6,
                'description': 'Routes authentication and login requests'
            },
            {
                'intent_name': 'general_support',
                'keywords': ['help', 'support', 'issue', 'problem', 'question'],
                'agent_id': support_agent['id'] if support_agent else None,
                'priority': 100,
                'confidence_threshold': 0.5,
                'description': 'Routes general support and help requests'
            }
        ]
        
        # Filter rules to only include those with valid agent_id
        valid_rules = [rule for rule in rules if rule['agent_id'] is not None]
        
        if valid_rules:
            rules_result = supabase.from_('router_rules').insert(valid_rules).execute()
            if rules_result.error:
                raise Exception(rules_result.error)
            print(f'✅ Inserted {len(valid_rules)} router rules')
        
        # Define fallback messages
        fallback_messages = [
            {
                'message': "I'm sorry, I didn't understand your request. Could you please rephrase it or ask for help?",
                'category': 'general',
                'is_active': True
            },
            {
                'message': "I'm not sure how to help with that. Would you like to speak with a support agent?",
                'category': 'general',
                'is_active': True
            },
            {
                'message': "That's an interesting question! Let me connect you with someone who can better assist you.",
                'category': 'general',
                'is_active': True
            },
            {
                'message': "I'm still learning! Could you try asking your question in a different way?",
                'category': 'general',
                'is_active': True
            },
            {
                'message': "I want to make sure I help you correctly. Could you provide more details about what you need?",
                'category': 'general',
                'is_active': True
            }
        ]
        
        fallback_result = supabase.from_('fallback_messages').insert(fallback_messages).execute()
        if fallback_result.error:
            raise Exception(fallback_result.error)
        print(f'✅ Inserted {len(fallback_messages)} fallback messages')
        
        print('🎉 Router rules and fallback messages seeded successfully!')
        
    except Exception as error:
        print(f'❌ Seeding failed: {error}')
        raise error

# Run seeding if called directly
if __name__ == '__main__':
    asyncio.run(seed_router_rules())
