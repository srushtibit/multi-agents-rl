"""
Email Configuration Helper for Escalation System

This utility helps users configure email settings for the escalation system.
"""

import os
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv, set_key

logger = logging.getLogger(__name__)

class EmailConfigHelper:
    """Helper class for managing email configuration."""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.env_path = ".env"

        # Load .env file if it exists
        if os.path.exists(self.env_path):
            load_dotenv(self.env_path)

        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _save_config(self) -> bool:
        """Save system configuration."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get_current_email_config(self) -> Dict[str, Any]:
        """Get current email configuration."""
        return self.config.get('email', {})
    
    def update_email_config(self,
                           smtp_server: str,
                           smtp_port: int,
                           use_tls: bool,
                           sender_email: str,
                           sender_password: str,
                           escalation_recipients: List[str]) -> bool:
        """
        Update email configuration.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port number
            use_tls: Whether to use TLS
            sender_email: Sender email address
            sender_password: Sender email password
            escalation_recipients: List of escalation recipient emails
            
        Returns:
            True if configuration was updated successfully
        """
        try:
            # Validate required fields
            if not smtp_server or not smtp_server.strip():
                logger.error("SMTP server cannot be empty")
                return False

            if not sender_email or not sender_email.strip():
                logger.error("Sender email cannot be empty")
                return False

            # Update configuration
            if 'email' not in self.config:
                self.config['email'] = {}

            self.config['email'].update({
                'smtp_server': smtp_server.strip(),
                'smtp_port': smtp_port,
                'use_tls': use_tls,
                'sender_email': sender_email.strip(),
                'sender_password': '${EMAIL_SENDER_PASSWORD}',  # Reference to env var
                'escalation_recipients': escalation_recipients
            })

            # Also update the .env file for the password
            success = self._save_config()
            if success:
                self._update_env_file(sender_password)

            return success

        except Exception as e:
            logger.error(f"Failed to update email config: {e}")
            return False

    def _update_env_file(self, sender_password: str):
        """Update the .env file with the email password."""
        try:
            # Ensure .env file exists
            if not os.path.exists(self.env_path):
                with open(self.env_path, 'w') as f:
                    f.write("# Environment Variables for NexaCorp Support System\n")

            # Update the EMAIL_SENDER_PASSWORD in .env file
            set_key(self.env_path, "EMAIL_SENDER_PASSWORD", sender_password)
            logger.info("Updated EMAIL_SENDER_PASSWORD in .env file")

        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
    
    def test_email_connection(self, 
                            smtp_server: str = None,
                            smtp_port: int = None,
                            use_tls: bool = None,
                            sender_email: str = None,
                            sender_password: str = None) -> Dict[str, Any]:
        """
        Test email connection with given or current configuration.
        
        Returns:
            Test result with status and details
        """
        # Use provided values or fall back to current config
        current_config = self.get_current_email_config()
        
        smtp_server = smtp_server or current_config.get('smtp_server')
        smtp_port = smtp_port or current_config.get('smtp_port', 587)
        use_tls = use_tls if use_tls is not None else current_config.get('use_tls', True)
        sender_email = sender_email or current_config.get('sender_email')
        sender_password = sender_password or current_config.get('sender_password')
        
        result = {
            'status': 'unknown',
            'details': {},
            'errors': []
        }
        
        # Check required fields
        if not smtp_server:
            result['errors'].append("SMTP server not specified")
        if not sender_email:
            result['errors'].append("Sender email not specified")
        if not sender_password:
            result['errors'].append("Sender password not specified")
        
        if result['errors']:
            result['status'] = 'configuration_error'
            return result
        
        # Test connection
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            if use_tls:
                server.starttls()
            
            server.login(sender_email, sender_password)
            server.quit()
            
            result['status'] = 'success'
            result['details'] = {
                'smtp_server': smtp_server,
                'smtp_port': smtp_port,
                'use_tls': use_tls,
                'sender_email': sender_email
            }
            
        except Exception as e:
            result['status'] = 'connection_error'
            result['errors'].append(str(e))
        
        return result
    
    def send_test_email(self, recipient: str, subject: str = None, body: str = None) -> bool:
        """
        Send a test email to verify configuration.
        
        Args:
            recipient: Test email recipient
            subject: Email subject (optional)
            body: Email body (optional)
            
        Returns:
            True if email was sent successfully
        """
        config = self.get_current_email_config()
        
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        use_tls = config.get('use_tls', True)
        sender_email = config.get('sender_email')
        sender_password = config.get('sender_password')
        
        if not all([smtp_server, sender_email, sender_password]):
            logger.error("Email configuration incomplete")
            return False
        
        try:
            # Create test message
            subject = subject or "🧪 Test Email - NexaCorp Support System"
            body = body or f"""
This is a test email from the NexaCorp Support System.

If you received this email, the escalation system is configured correctly.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: NexaCorp AI Support System
Component: Escalation Agent

Best regards,
NexaCorp Support System
            """.strip()
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = recipient
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            if use_tls:
                server.starttls()
            
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Test email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send test email: {e}")
            return False
    
    def get_common_smtp_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get common SMTP configurations for popular email providers."""
        return {
            "Gmail": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "notes": "Use app-specific password, not regular password"
            },
            "Outlook/Hotmail": {
                "smtp_server": "smtp-mail.outlook.com",
                "smtp_port": 587,
                "use_tls": True,
                "notes": "May require app password for 2FA accounts"
            },
            "Yahoo": {
                "smtp_server": "smtp.mail.yahoo.com",
                "smtp_port": 587,
                "use_tls": True,
                "notes": "Requires app password"
            },
            "Custom SMTP": {
                "smtp_server": "mail.yourcompany.com",
                "smtp_port": 587,
                "use_tls": True,
                "notes": "Contact your IT administrator for settings"
            }
        }
    
    def validate_email_address(self, email: str) -> bool:
        """
        Basic email address validation.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email appears valid
        """
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def get_setup_instructions(self) -> Dict[str, str]:
        """Get setup instructions for different email providers."""
        return {
            "Gmail": """
1. Enable 2-Factor Authentication on your Google account
2. Go to Google Account settings > Security > App passwords
3. Generate an app-specific password for 'Mail'
4. Use this app password instead of your regular password
5. SMTP Server: smtp.gmail.com, Port: 587, TLS: Yes
            """.strip(),
            
            "Outlook": """
1. Go to Outlook.com > Settings > Mail > Sync email
2. Enable IMAP access
3. For 2FA accounts, create an app password
4. SMTP Server: smtp-mail.outlook.com, Port: 587, TLS: Yes
            """.strip(),
            
            "Corporate Email": """
1. Contact your IT administrator for SMTP settings
2. Common settings: Port 587 or 25, TLS enabled
3. May require VPN connection for external access
4. Some systems use authentication different from email password
            """.strip()
        }

# Convenience functions for direct use
def get_email_helper() -> EmailConfigHelper:
    """Get email configuration helper instance."""
    return EmailConfigHelper()

def test_current_email_config() -> Dict[str, Any]:
    """Test current email configuration."""
    helper = EmailConfigHelper()
    return helper.test_email_connection()

def send_test_email_to(recipient: str) -> bool:
    """Send test email to specified recipient."""
    helper = EmailConfigHelper()
    return helper.send_test_email(recipient)
