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
            elif "21608" in error_message:
                details["suggestion"] = "The 'To' phone number is not a valid phone number format."
            elif "20003" in error_message:
                details["suggestion"] = "Authentication error. Check your Twilio Account SID and Auth Token."
            return False, details


# Singleton instance
sms_service = SmsService()