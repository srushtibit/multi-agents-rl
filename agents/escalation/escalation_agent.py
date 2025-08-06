"""
Escalation Agent for the multilingual multi-agent support system.
Handles severity detection and automated email escalation for critical issues.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from agents.base_agent import BaseAgent, Message, MessageType
from utils.language_utils import detect_language, translate_to_english
from utils.config_loader import get_config

logger = logging.getLogger(__name__)

@dataclass
class SeverityAssessment:
    """Result of severity assessment."""
    severity_level: str  # 'low', 'medium', 'high', 'critical'
    severity_score: float  # 0.0 to 1.0
    trigger_keywords: List[str]
    reasoning: str
    requires_escalation: bool
    urgency_indicators: List[str]

@dataclass
class EscalationRecord:
    """Record of an escalation action."""
    escalation_id: str
    original_message_id: str
    severity_assessment: SeverityAssessment
    escalation_timestamp: str
    email_sent: bool
    email_recipients: List[str]
    email_subject: str
    email_body: str
    follow_up_required: bool
    resolution_deadline: Optional[str]

class EscalationAgent(BaseAgent):
    """Agent responsible for detecting high-severity issues and escalating them."""
    
    def __init__(self, agent_id: str = "escalation_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Configuration
        self.severity_threshold = self.system_config.get('agents.escalation.severity_threshold', 0.9)
        self.email_delay = self.system_config.get('agents.escalation.email_delay', 300)  # 5 minutes
        
        # Email configuration
        email_config = self.system_config.get('email', {})
        self.smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = email_config.get('smtp_port', 587)
        self.use_tls = email_config.get('use_tls', True)
        self.sender_email = email_config.get('sender_email', '')
        self.sender_password = email_config.get('sender_password', '')
        self.recipients = email_config.get('escalation_recipients', [])
        
        # Severity keywords from configuration
        escalation_config = self.system_config.get('agents.escalation', {})
        self.high_severity_keywords = escalation_config.get('keywords', {}).get('high_severity', [
            'urgent', 'critical', 'emergency', 'lawsuit', 'legal', 'security breach', 
            'data loss', 'system down', 'outage', 'breach', 'hack', 'attack',
            'cannot access', 'locked out', 'deadline', 'crisis', 'immediate'
        ])
        
        self.medium_severity_keywords = escalation_config.get('keywords', {}).get('medium_severity', [
            'important', 'asap', 'priority', 'deadline', 'urgent', 'soon',
            'problem', 'issue', 'error', 'fault', 'broken', 'not working'
        ])
        
        # Urgency indicators
        self.urgency_patterns = {
            'time_sensitive': [
                r'within \d+ (hours?|minutes?|days?)',
                r'by (today|tomorrow|end of day|eod)',
                r'deadline',
                r'expires?',
                r'time limit'
            ],
            'business_critical': [
                r'production (down|issue|problem)',
                r'(all|entire) (team|department|company)',
                r'customer(s)? (affected|impacted)',
                r'revenue (loss|impact)',
                r'business (critical|impact)'
            ],
            'security_related': [
                r'(password|account) (compromised|hacked)',
                r'(data|information) (breach|leak)',
                r'(security|safety) (concern|issue)',
                r'(unauthorized|suspicious) (access|activity)',
                r'(malware|virus|attack)'
            ],
            'legal_compliance': [
                r'(legal|lawsuit|litigation)',
                r'(compliance|regulation|audit)',
                r'(gdpr|hipaa|sox)',
                r'(lawsuit|legal action)',
                r'(regulatory|government)'
            ]
        }
        
        # Escalation tracking
        self.escalation_history: List[EscalationRecord] = []
        self.pending_escalations: Dict[str, EscalationRecord] = {}
        
        # Statistics
        self.escalation_stats = {
            'total_assessments': 0,
            'escalations_triggered': 0,
            'emails_sent': 0,
            'false_positives': 0,
            'severity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        }
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [
            "severity_assessment",
            "escalation_detection",
            "email_notification",
            "urgency_analysis",
            "multilingual_severity_detection",
            "automated_escalation"
        ]
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process message to assess severity and handle escalation.
        
        Args:
            message: Message to assess for escalation
            
        Returns:
            Optional escalation notification message
        """
        try:
            # Assess severity
            severity_assessment = await self._assess_severity(message)
            
            # Update statistics
            self.escalation_stats['total_assessments'] += 1
            self.escalation_stats['severity_distribution'][severity_assessment.severity_level] += 1
            
            # Handle escalation if required
            response_message = None
            if severity_assessment.requires_escalation:
                response_message = await self._handle_escalation(message, severity_assessment)
            
            # Log assessment
            self._log_action("severity_assessment", {
                "severity_level": severity_assessment.severity_level,
                "severity_score": severity_assessment.severity_score,
                "requires_escalation": severity_assessment.requires_escalation,
                "trigger_keywords": severity_assessment.trigger_keywords
            })
            
            return response_message
            
        except Exception as e:
            logger.error(f"Error in escalation agent: {e}")
            self._log_action("escalation_error", {"error": str(e)}, success=False, error_message=str(e))
            return None
    
    async def _assess_severity(self, message: Message) -> SeverityAssessment:
        """
        Assess the severity of a message.
        
        Args:
            message: Message to assess
            
        Returns:
            Severity assessment result
        """
        content = message.content.lower()
        original_content = message.content
        
        # Translate to English if needed for better keyword matching
        if message.language != 'en':
            try:
                translation = translate_to_english(message.content, message.language)
                if translation.confidence > 0.7:
                    content = translation.translated_text.lower()
            except Exception as e:
                logger.warning(f"Translation failed for severity assessment: {e}")
        
        # Initialize assessment
        severity_score = 0.0
        trigger_keywords = []
        urgency_indicators = []
        reasoning_parts = []
        
        # Check for high severity keywords
        high_severity_matches = []
        for keyword in self.high_severity_keywords:
            if keyword.lower() in content:
                high_severity_matches.append(keyword)
                trigger_keywords.append(keyword)
                severity_score += 0.3
        
        if high_severity_matches:
            reasoning_parts.append(f"High severity keywords detected: {', '.join(high_severity_matches)}")
        
        # Check for medium severity keywords
        medium_severity_matches = []
        for keyword in self.medium_severity_keywords:
            if keyword.lower() in content:
                medium_severity_matches.append(keyword)
                trigger_keywords.append(keyword)
                severity_score += 0.1
        
        if medium_severity_matches:
            reasoning_parts.append(f"Medium severity keywords detected: {', '.join(medium_severity_matches)}")
        
        # Check urgency patterns
        for category, patterns in self.urgency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    urgency_indicators.append(f"{category}: {pattern}")
                    severity_score += 0.2
                    reasoning_parts.append(f"Urgency pattern detected ({category}): {pattern}")
        
        # Additional severity indicators
        
        # Repetition/emphasis (caps, exclamation marks)
        caps_ratio = sum(1 for c in original_content if c.isupper()) / max(len(original_content), 1)
        if caps_ratio > 0.3:
            severity_score += 0.1
            reasoning_parts.append("Excessive capitalization detected")
        
        exclamation_count = original_content.count('!')
        if exclamation_count > 2:
            severity_score += 0.05 * min(exclamation_count, 10)
            reasoning_parts.append(f"Multiple exclamation marks ({exclamation_count})")
        
        # Time sensitivity indicators
        time_words = ['immediate', 'now', 'asap', 'urgent', 'emergency']
        time_matches = sum(1 for word in time_words if word in content)
        if time_matches > 0:
            severity_score += 0.1 * time_matches
            reasoning_parts.append(f"Time sensitivity indicators: {time_matches}")
        
        # Scope indicators (affecting multiple people/systems)
        scope_words = ['all', 'entire', 'everyone', 'company', 'department', 'team', 'system']
        scope_matches = sum(1 for word in scope_words if word in content)
        if scope_matches > 1:
            severity_score += 0.1
            reasoning_parts.append("Wide scope impact indicated")
        
        # Business impact indicators
        business_words = ['revenue', 'customer', 'client', 'sales', 'production', 'business']
        business_matches = sum(1 for word in business_words if word in content)
        if business_matches > 0:
            severity_score += 0.05 * business_matches
            reasoning_parts.append(f"Business impact indicators: {business_matches}")
        
        # Determine severity level
        if severity_score >= 0.9:
            severity_level = 'critical'
        elif severity_score >= 0.7:
            severity_level = 'high'
        elif severity_score >= 0.4:
            severity_level = 'medium'
        else:
            severity_level = 'low'
        
        # Determine if escalation is required
        requires_escalation = severity_score >= self.severity_threshold
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No significant severity indicators detected"
        
        return SeverityAssessment(
            severity_level=severity_level,
            severity_score=min(1.0, severity_score),
            trigger_keywords=trigger_keywords,
            reasoning=reasoning,
            requires_escalation=requires_escalation,
            urgency_indicators=urgency_indicators
        )
    
    async def _handle_escalation(self, message: Message, severity_assessment: SeverityAssessment) -> Message:
        """
        Handle escalation process.
        
        Args:
            message: Original message
            severity_assessment: Severity assessment result
            
        Returns:
            Escalation notification message
        """
        try:
            # Create escalation record
            escalation_id = f"ESC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{message.id[:8]}"
            
            # Prepare email content
            email_subject, email_body = self._prepare_escalation_email(message, severity_assessment, escalation_id)
            
            # Create escalation record
            escalation_record = EscalationRecord(
                escalation_id=escalation_id,
                original_message_id=message.id,
                severity_assessment=severity_assessment,
                escalation_timestamp=datetime.now().isoformat(),
                email_sent=False,
                email_recipients=self.recipients.copy(),
                email_subject=email_subject,
                email_body=email_body,
                follow_up_required=severity_assessment.severity_level in ['high', 'critical'],
                resolution_deadline=self._calculate_resolution_deadline(severity_assessment.severity_level)
            )
            
            # Send email notification
            email_sent = await self._send_escalation_email(escalation_record)
            escalation_record.email_sent = email_sent
            
            # Store escalation record
            self.escalation_history.append(escalation_record)
            if escalation_record.follow_up_required:
                self.pending_escalations[escalation_id] = escalation_record
            
            # Update statistics
            self.escalation_stats['escalations_triggered'] += 1
            if email_sent:
                self.escalation_stats['emails_sent'] += 1
            
            # Create response message
            response_content = f"ESCALATION TRIGGERED: {severity_assessment.severity_level.upper()} severity issue detected.\n"
            response_content += f"Escalation ID: {escalation_id}\n"
            response_content += f"Severity Score: {severity_assessment.severity_score:.3f}\n"
            response_content += f"Email Notification: {'Sent' if email_sent else 'Failed'}\n"
            response_content += f"Reason: {severity_assessment.reasoning}"
            
            response_message = Message(
                type=MessageType.ESCALATION,
                content=response_content,
                metadata={
                    'escalation_id': escalation_id,
                    'severity_assessment': {
                        'level': severity_assessment.severity_level,
                        'score': severity_assessment.severity_score,
                        'keywords': severity_assessment.trigger_keywords,
                        'reasoning': severity_assessment.reasoning
                    },
                    'email_sent': email_sent,
                    'resolution_deadline': escalation_record.resolution_deadline
                },
                sender=self.agent_id,
                recipient="broadcast"  # Notify all agents
            )
            
            logger.warning(f"ESCALATION: {severity_assessment.severity_level} severity issue escalated - {escalation_id}")
            
            return response_message
            
        except Exception as e:
            logger.error(f"Error handling escalation: {e}")
            raise
    
    def _prepare_escalation_email(self, 
                                message: Message, 
                                severity_assessment: SeverityAssessment,
                                escalation_id: str) -> Tuple[str, str]:
        """
        Prepare email subject and body for escalation.
        
        Args:
            message: Original message
            severity_assessment: Severity assessment
            escalation_id: Escalation identifier
            
        Returns:
            Tuple of (subject, body)
        """
        # Email subject
        subject = f"🚨 {severity_assessment.severity_level.upper()} SEVERITY ALERT - {escalation_id}"
        
        # Email body
        body_parts = [
            "AUTOMATED ESCALATION ALERT",
            "=" * 50,
            "",
            f"Escalation ID: {escalation_id}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Severity Level: {severity_assessment.severity_level.upper()}",
            f"Severity Score: {severity_assessment.severity_score:.3f}/1.0",
            "",
            "ISSUE DETAILS:",
            "-" * 20,
            f"Original Message ID: {message.id}",
            f"Message Language: {message.language}",
            f"Message Timestamp: {message.timestamp}",
            "",
            "MESSAGE CONTENT:",
            "-" * 20,
            message.content,
            "",
            "SEVERITY ANALYSIS:",
            "-" * 20,
            f"Reasoning: {severity_assessment.reasoning}",
            "",
            "TRIGGER KEYWORDS:",
            ", ".join(severity_assessment.trigger_keywords) if severity_assessment.trigger_keywords else "None",
            "",
            "URGENCY INDICATORS:",
            "\n".join(f"- {indicator}" for indicator in severity_assessment.urgency_indicators) if severity_assessment.urgency_indicators else "None",
            "",
            "RECOMMENDED ACTIONS:",
            "-" * 20
        ]
        
        # Add severity-specific recommendations
        if severity_assessment.severity_level == 'critical':
            body_parts.extend([
                "🔴 IMMEDIATE ACTION REQUIRED",
                "- Respond within 15 minutes",
                "- Escalate to on-call manager",
                "- Consider emergency procedures",
                "- Document all actions taken"
            ])
        elif severity_assessment.severity_level == 'high':
            body_parts.extend([
                "🟠 URGENT ACTION REQUIRED",
                "- Respond within 1 hour", 
                "- Assign senior support staff",
                "- Monitor for escalation",
                "- Keep stakeholders informed"
            ])
        else:
            body_parts.extend([
                "🟡 PROMPT ACTION REQUIRED",
                "- Respond within 4 hours",
                "- Assign appropriate support staff",
                "- Follow standard procedures"
            ])
        
        body_parts.extend([
            "",
            "SYSTEM INFORMATION:",
            "-" * 20,
            f"System: NexaCorp AI Support System",
            f"Agent: Escalation Agent",
            f"Environment: {self.system_config.get('system.environment', 'Unknown')}",
            "",
            "This is an automated alert. Please respond promptly.",
            "",
            "Best regards,",
            "NexaCorp AI Support System"
        ])
        
        body = "\n".join(body_parts)
        return subject, body
    
    async def _send_escalation_email(self, escalation_record: EscalationRecord) -> bool:
        """
        Send escalation email notification.
        
        Args:
            escalation_record: Escalation record with email details
            
        Returns:
            True if email was sent successfully
        """
        if not self.sender_email or not self.sender_password or not self.recipients:
            logger.warning("Email configuration incomplete, cannot send escalation email")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(escalation_record.email_recipients)
            msg['Subject'] = escalation_record.email_subject
            
            # Add body
            msg.attach(MIMEText(escalation_record.email_body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, escalation_record.email_recipients, text)
            server.quit()
            
            logger.info(f"Escalation email sent successfully for {escalation_record.escalation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send escalation email: {e}")
            return False
    
    def _calculate_resolution_deadline(self, severity_level: str) -> str:
        """Calculate resolution deadline based on severity level."""
        now = datetime.now()
        
        if severity_level == 'critical':
            deadline = now + timedelta(hours=1)
        elif severity_level == 'high':
            deadline = now + timedelta(hours=4)
        elif severity_level == 'medium':
            deadline = now + timedelta(hours=24)
        else:
            deadline = now + timedelta(hours=72)
        
        return deadline.isoformat()
    
    def get_escalation_stats(self) -> Dict[str, Any]:
        """Get escalation statistics."""
        stats = self.escalation_stats.copy()
        
        # Add calculated metrics
        if stats['total_assessments'] > 0:
            stats['escalation_rate'] = stats['escalations_triggered'] / stats['total_assessments']
        else:
            stats['escalation_rate'] = 0.0
        
        if stats['escalations_triggered'] > 0:
            stats['email_success_rate'] = stats['emails_sent'] / stats['escalations_triggered']
        else:
            stats['email_success_rate'] = 0.0
        
        # Recent escalations
        recent_escalations = [e for e in self.escalation_history[-20:]]  # Last 20
        stats['recent_escalations'] = [
            {
                'id': e.escalation_id,
                'timestamp': e.escalation_timestamp,
                'severity': e.severity_assessment.severity_level,
                'email_sent': e.email_sent
            }
            for e in recent_escalations
        ]
        
        # Pending escalations
        stats['pending_escalations'] = len(self.pending_escalations)
        
        return stats
    
    def get_escalation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get escalation history."""
        recent_escalations = self.escalation_history[-limit:]
        return [
            {
                'escalation_id': e.escalation_id,
                'original_message_id': e.original_message_id,
                'severity_level': e.severity_assessment.severity_level,
                'severity_score': e.severity_assessment.severity_score,
                'timestamp': e.escalation_timestamp,
                'email_sent': e.email_sent,
                'recipients': e.email_recipients,
                'reasoning': e.severity_assessment.reasoning,
                'trigger_keywords': e.severity_assessment.trigger_keywords
            }
            for e in recent_escalations
        ]
    
    def mark_escalation_resolved(self, escalation_id: str) -> bool:
        """
        Mark an escalation as resolved.
        
        Args:
            escalation_id: ID of the escalation to resolve
            
        Returns:
            True if escalation was found and marked resolved
        """
        if escalation_id in self.pending_escalations:
            del self.pending_escalations[escalation_id]
            self._log_action("escalation_resolved", {"escalation_id": escalation_id})
            logger.info(f"Escalation {escalation_id} marked as resolved")
            return True
        return False
    
    def test_email_configuration(self) -> Dict[str, Any]:
        """
        Test email configuration.
        
        Returns:
            Test result with status and details
        """
        result = {
            'status': 'unknown',
            'details': {},
            'errors': []
        }
        
        # Check configuration
        if not self.sender_email:
            result['errors'].append("Sender email not configured")
        
        if not self.sender_password:
            result['errors'].append("Sender password not configured")
        
        if not self.recipients:
            result['errors'].append("No escalation recipients configured")
        
        if result['errors']:
            result['status'] = 'configuration_error'
            return result
        
        # Test SMTP connection
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            
            server.login(self.sender_email, self.sender_password)
            server.quit()
            
            result['status'] = 'success'
            result['details'] = {
                'smtp_server': self.smtp_server,
                'smtp_port': self.smtp_port,
                'use_tls': self.use_tls,
                'sender_email': self.sender_email,
                'recipients_count': len(self.recipients)
            }
            
        except Exception as e:
            result['status'] = 'connection_error'
            result['errors'].append(str(e))
        
        return result