"""
SIM Swap Agent - Finite State Machine Implementation

Handles SIM swap requests through a multi-step process:
1. collect_user_details - Gather customer information
2. verify_identity - Security verification
3. confirm_swap - Process and confirm the swap
4. workflow_complete - End the process
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time

class SimSwapAgent:
    """SIM Swap Agent with FSM-based workflow for secure SIM card swapping"""
    
    @classmethod
    async def handle_sim_swap_step(cls, state: Dict[str, Any] = None, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a single step in the SIM swap process
        
        Args:
            state: Current FSM state
            input_data: User input data
            
        Returns:
            Response with next state and intent
        """
        if state is None:
            state = {}
        if input_data is None:
            input_data = {}
            
        current_step = state.get('step', 'initial')
        context = state.get('context', {})
        
        if current_step in ['initial', 'collect_user_details']:
            return cls.handle_collect_user_details(input_data, context)
        elif current_step == 'verify_identity':
            return cls.handle_verify_identity(input_data, context)
        elif current_step == 'confirm_swap':
            return cls.handle_confirm_swap(input_data, context)
        elif current_step == 'workflow_complete':
            return cls.handle_complete(input_data, context)
        else:
            return cls.handle_initial(input_data, context)
    
    @classmethod
    def handle_initial(cls, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Initial step - start the SIM swap process"""
        return {
            "response": {
                "message": "Welcome to SIM Swap Service. I'll help you swap your SIM card safely.",
                "instructions": "To proceed, I need to collect some details from you.",
                "step": "collect_user_details"
            },
            "output_intent": "collect_user_details",
            "newState": {
                "step": "collect_user_details",
                "context": {
                    **context,
                    "started_at": datetime.now().isoformat(),
                    "request_id": f"sim_swap_{int(time.time() * 1000)}"
                }
            }
        }
    
    @classmethod
    def handle_collect_user_details(cls, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Collect user details"""
        # Simulate collecting user details
        user_details = {
            "mobile_number": input_data.get('mobile_number') or input_data.get('phone', "1234567890"),
            "full_name": input_data.get('full_name') or input_data.get('name', "krishna"),
            "account_number": input_data.get('account_number', "ACC123456"),
            "current_sim_number": input_data.get('current_sim_number', "SIM789012")
        }
        
        return {
            "response": {
                "message": "Thank you for providing your details. Now I need to verify your identity for security purposes.",
                "collected_details": user_details,
                "next_step": "Identity verification required",
                "security_notice": "For your protection, we need to verify your identity before proceeding."
            },
            "output_intent": "verify_identity",
            "newState": {
                "step": "verify_identity",
                "context": {
                    **context,
                    "user_details": user_details,
                    "details_collected_at": datetime.now().isoformat()
                }
            }
        }
    
    @classmethod
    def handle_verify_identity(cls, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Verify identity"""
        # Simulate identity verification process
        verification_methods = [
            "Security question verification",
            "SMS OTP to registered number",
            "Document verification"
        ]
        
        security_question = "What is your mother's maiden name?"
        provided_answer = input_data.get('security_answer') or input_data.get('answer', "Smith")
        
        # Mock verification logic
        is_verified = len(provided_answer) > 0  # Simple mock verification
        
        if is_verified:
            return {
                "response": {
                    "message": "Identity verification successful! Proceeding with SIM swap confirmation.",
                    "verification_status": "VERIFIED",
                    "verification_method": "Security Question",
                    "next_step": "SIM swap confirmation"
                },
                "output_intent": "confirm_swap",
                "newState": {
                    "step": "confirm_swap",
                    "context": {
                        **context,
                        "verification_status": "VERIFIED",
                        "verified_at": datetime.now().isoformat(),
                        "verification_method": "security_question"
                    }
                }
            }
        else:
            return {
                "response": {
                    "message": "Please answer the security question to verify your identity.",
                    "security_question": security_question,
                    "available_methods": verification_methods,
                    "retry_count": context.get('retry_count', 0) + 1
                },
                "output_intent": "verify_identity",
                "newState": {
                    "step": "verify_identity",
                    "context": {
                        **context,
                        "retry_count": context.get('retry_count', 0) + 1,
                        "last_attempt_at": datetime.now().isoformat()
                    }
                }
            }
    
    @classmethod
    def handle_confirm_swap(cls, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Confirm SIM swap"""
        # Generate mock SIM swap details
        current_time = int(time.time() * 1000)
        estimated_completion = datetime.now() + timedelta(hours=2)
        
        user_details = context.get('user_details', {})
        swap_details = {
            "old_sim_number": user_details.get('current_sim_number', "SIM789012"),
            "new_sim_number": f"SIM{str(current_time)[-6:]}",
            "swap_reference": f"SWAP_{current_time}",
            "estimated_completion": estimated_completion.isoformat(),
            "activation_instructions": [
                "Insert the new SIM card into your device",
                "Restart your device",
                "Wait for network activation (up to 2 hours)",
                "Test by making a call"
            ]
        }
        
        return {
            "response": {
                "message": "SIM swap request has been processed successfully!",
                "swap_details": swap_details,
                "status": "CONFIRMED",
                "important_notes": [
                    "Your old SIM will be deactivated within 2 hours",
                    "Keep your device on during the activation process",
                    "Contact support if you face any issues"
                ],
                "support_contact": "1-800-SIM-HELP"
            },
            "output_intent": "workflow_complete",
            "newState": {
                "step": "workflow_complete",
                "context": {
                    **context,
                    "swap_details": swap_details,
                    "confirmed_at": datetime.now().isoformat(),
                    "status": "CONFIRMED"
                }
            }
        }
    
    @classmethod
    def handle_complete(cls, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Final step: Process complete"""
        swap_details = context.get('swap_details', {})
        
        return {
            "response": {
                "message": "SIM swap process completed successfully. Thank you for using our service!",
                "summary": {
                    "request_id": context.get('request_id'),
                    "status": "COMPLETED",
                    "processing_time": cls.calculate_processing_time(context.get('started_at')),
                    "reference_number": swap_details.get('swap_reference')
                },
                "next_steps": [
                    "Monitor your device for network activation",
                    "Contact support if needed",
                    "Dispose of old SIM card securely"
                ]
            },
            "output_intent": "workflow_complete",
            "newState": {
                "step": "workflow_complete",
                "context": {
                    **context,
                    "completed_at": datetime.now().isoformat(),
                    "final_status": "COMPLETED"
                }
            }
        }
    
    @classmethod
    def calculate_processing_time(cls, start_time: Optional[str]) -> str:
        """Calculate processing time for summary"""
        if not start_time:
            return "Unknown"
        
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            now = datetime.now()
            diff_ms = (now - start).total_seconds() * 1000
            diff_mins = int(diff_ms / 60000)
            
            if diff_mins < 1:
                return "Less than 1 minute"
            elif diff_mins == 1:
                return "1 minute"
            else:
                return f"{diff_mins} minutes"
        except Exception:
            return "Unknown"
    
    @classmethod
    def get_available_transitions(cls, current_step: str) -> List[str]:
        """Get available transitions from current state"""
        transitions = {
            'initial': ['collect_user_details'],
            'collect_user_details': ['verify_identity'],
            'verify_identity': ['confirm_swap', 'verify_identity'],  # Can retry verification
            'confirm_swap': ['workflow_complete'],
            'workflow_complete': []
        }
        
        return transitions.get(current_step, [])
    
    @classmethod
    def validate_step_input(cls, step: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input for current step"""
        validations = {
            'collect_user_details': ['mobile_number', 'full_name'],
            'verify_identity': ['security_answer'],
            'confirm_swap': [],  # No specific input required
            'workflow_complete': []
        }
        
        required = validations.get(step, [])
        missing = [field for field in required if not input_data.get(field)]
        
        return {
            "isValid": len(missing) == 0,
            "missingFields": missing,
            "message": f"Missing required fields: {', '.join(missing)}" if missing else "Valid"
        }


# Export the main handler function for backward compatibility
async def handle_sim_swap_step(state: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility function"""
    return await SimSwapAgent.handle_sim_swap_step(state, input_data)
