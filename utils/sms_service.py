"""
SMS notification service for the CIVILIAN platform.
Provides functionality to send SMS messages about critical misinformation alerts.
"""
import os
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Check if Twilio is available
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    logger.warning("Twilio package not installed. SMS notifications will be disabled.")
    TWILIO_AVAILABLE = False


class SmsService:
    """Service for sending SMS notifications via Twilio."""
    
    def __init__(self):
        """Initialize the SMS service with Twilio credentials."""
        # Initialize Twilio configuration
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER")
        self.recipient_phone = os.environ.get("RECIPIENT_PHONE_NUMBER")
        
        # Check if Twilio is properly configured
        self.is_configured = all([
            TWILIO_AVAILABLE,
            self.account_sid,
            self.auth_token,
            self.twilio_phone,
            self.recipient_phone
        ])
        
        # Initialize Twilio client if configured
        self.client = None
        if self.is_configured:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Twilio SMS service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                self.is_configured = False
    
    def verify_configuration(self) -> dict:
        """
        Verify the Twilio configuration and check common issues.
        
        Returns:
            Dictionary with verification results and suggestions
        """
        results = {
            "configured": self.is_configured,
            "client_initialized": self.client is not None,
            "suggestions": [],
            "is_trial_account": False
        }
        
        # Check if Twilio package is installed
        if not TWILIO_AVAILABLE:
            results["suggestions"].append("Twilio package is not installed. Install with: pip install twilio")
            
        # Check credentials
        if not self.account_sid:
            results["suggestions"].append("TWILIO_ACCOUNT_SID environment variable is not set")
        if not self.auth_token:
            results["suggestions"].append("TWILIO_AUTH_TOKEN environment variable is not set")
            
        # Check phone numbers
        if not self.twilio_phone:
            results["suggestions"].append("TWILIO_PHONE_NUMBER environment variable is not set")
        elif not self.twilio_phone.startswith('+'):
            results["suggestions"].append("TWILIO_PHONE_NUMBER should start with '+' followed by country code")
            
        if not self.recipient_phone:
            results["suggestions"].append("RECIPIENT_PHONE_NUMBER environment variable is not set")
        elif not self.recipient_phone.startswith('+'):
            results["suggestions"].append("RECIPIENT_PHONE_NUMBER should start with '+' followed by country code")
            
        # Check if it's a trial account
        if self.twilio_phone and (self.twilio_phone.startswith('+1620') or 
                                 self.twilio_phone.startswith('+1415') or 
                                 self.twilio_phone.startswith('+1843')):
            results["is_trial_account"] = True
            results["suggestions"].append(
                "You appear to be using a Twilio trial account. Verify that your recipient number is verified in your Twilio dashboard."
            )
            
        return results
            
    def send_message(self, message: str, recipient: Optional[str] = None) -> tuple[bool, dict]:
        """
        Send an SMS message via Twilio.
        
        Args:
            message: The text message to send
            recipient: Optional override for the default recipient phone number
            
        Returns:
            Tuple with (success_bool, details_dict)
        """
        # Create details dictionary with masked phone numbers for privacy
        details = {
            "configured": self.is_configured,
            "client_initialized": self.client is not None,
            "twilio_phone": self.twilio_phone
        }
        
        # Safely add masked recipient phone if available
        recipient_number = recipient or self.recipient_phone
        if recipient_number and len(recipient_number) >= 4:
            details["recipient_phone"] = "XXXX" + recipient_number[-4:]
        else:
            details["recipient_phone"] = "Unknown"
        
        # Check if service is configured
        if not self.is_configured or not self.client:
            logger.warning("SMS service not configured, cannot send message")
            details["error"] = "SMS service not configured"
            details["verification_results"] = self.verify_configuration()
            return False, details
        
        # Use default recipient if none provided
        to_number = recipient or self.recipient_phone
        
        try:
            # Send the message
            sms = self.client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=to_number
            )
            logger.info(f"SMS sent successfully. SID: {sms.sid}")
            details["sid"] = sms.sid
            details["status"] = sms.status
            return True, details
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to send SMS: {error_message}")
            details["error"] = error_message
            
            # Check for common Twilio errors
            if "21612" in error_message:
                details["suggestion"] = "The 'To' or 'From' phone numbers are not properly configured. For trial accounts, verify numbers are confirmed in your Twilio dashboard."
                details["error_code"] = "21612"
                details["verification_steps"] = [
                    "1. Log in to your Twilio Console",
                    "2. Navigate to 'Phone Numbers' > 'Verified Caller IDs'",
                    "3. Verify that your recipient number is on the list",
                    "4. If not, add and verify it by following Twilio's verification process"
                ]
            elif "21608" in error_message:
                details["suggestion"] = "The 'To' phone number is not a valid phone number format."
                details["error_code"] = "21608"
                details["verification_steps"] = [
                    "1. Ensure the phone number uses the E.164 format",
                    "2. Include the country code with a '+' prefix, e.g., +1XXXXXXXXXX for US numbers",
                    "3. Remove any spaces, dashes, or parentheses"
                ]
            elif "20003" in error_message:
                details["suggestion"] = "Authentication error. Check your Twilio Account SID and Auth Token."
                details["error_code"] = "20003"
                details["verification_steps"] = [
                    "1. Log in to your Twilio Console",
                    "2. Verify your Account SID and Auth Token",
                    "3. Ensure you're using the most recent Auth Token if you've reset it"
                ]
            else:
                details["suggestion"] = "An unknown error occurred. Check the error message for details."
                details["verification_steps"] = [
                    "1. Check your Twilio Console for more information", 
                    "2. Verify your account has sufficient funds if not a trial account",
                    "3. Ensure your Twilio account is active and not suspended"
                ]
                
            # Add verification results
            details["verification_results"] = self.verify_configuration()
            return False, details


# Singleton instance
sms_service = SmsService()