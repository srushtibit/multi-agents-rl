"""
Configuration loader utility for the multilingual multi-agent support system.
Handles loading and validation of system configuration from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System configuration dataclass with validation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        # Validate required sections
        required_sections = ['system', 'languages', 'llm', 'knowledge_base', 'agents']
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required configuration section: {section}")
        
        self.config = config_dict
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate language codes
        supported_langs = self.config['languages']['supported']
        if not isinstance(supported_langs, list) or len(supported_langs) == 0:
            raise ValueError("Supported languages must be a non-empty list")
        
        # Validate thresholds
        if not 0 <= self.config['knowledge_base']['similarity_threshold'] <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        if not 0 <= self.config['agents']['escalation']['severity_threshold'] <= 1:
            raise ValueError("Severity threshold must be between 0 and 1")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'agents.communication.learning_rate')."""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config_ref = self.config
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        config_ref[keys[-1]] = value

class ConfigLoader:
    """Configuration loader with environment variable support and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/system_config.yaml"
        self.config: Optional[SystemConfig] = None
        self._env_prefix = "NEXACORP_"
    
    def load_config(self, override_path: Optional[str] = None) -> SystemConfig:
        """Load configuration from YAML file with environment variable overrides."""
        config_file = override_path or self.config_path
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Apply environment variable overrides
            self._apply_env_overrides(config_data)
            
            self.config = SystemConfig(config_data)
            logger.info(f"Configuration loaded successfully from {config_file}")
            return self.config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]):
        """Apply environment variable overrides to configuration."""
        # Define environment variable mappings
        env_mappings = {
            f"{self._env_prefix}DEBUG": "system.debug",
            f"{self._env_prefix}LOG_LEVEL": "system.log_level",
            f"{self._env_prefix}OLLAMA_URL": "llm.ollama.base_url",
            f"{self._env_prefix}SMTP_SERVER": "email.smtp_server",
            f"{self._env_prefix}SMTP_PORT": "email.smtp_port",
            f"{self._env_prefix}SENDER_EMAIL": "email.sender_email",
            f"{self._env_prefix}SENDER_PASSWORD": "email.sender_password",
            f"{self._env_prefix}DB_CONNECTION": "database.connection_string",
            f"{self._env_prefix}API_PORT": "api.port",
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_data, config_path, self._convert_env_value(env_value))
    
    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: Any):
        """Set a nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Convert boolean values
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def save_config(self, config: SystemConfig, output_path: Optional[str] = None):
        """Save configuration to YAML file."""
        output_file = output_path or self.config_path
        
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config(self) -> SystemConfig:
        """Get the loaded configuration."""
        if self.config is None:
            self.load_config()
        return self.config

# Global configuration loader instance
config_loader = ConfigLoader()

def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config_loader.get_config()

def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from file."""
    return config_loader.load_config(config_path)

# Configuration validation utilities
def validate_language_code(lang_code: str) -> bool:
    """Validate if language code is supported."""
    config = get_config()
    return lang_code in config.get('languages.supported', [])

def validate_model_availability(model_name: str) -> bool:
    """Validate if LLM model is available."""
    # This would typically check with the model provider
    # For now, return True for all models
    return True

def get_email_template_path(template_name: str) -> str:
    """Get path to email template."""
    config = get_config()
    templates_path = config.get('email.templates_path', 'config/email_templates')
    return os.path.join(templates_path, f"{template_name}.html")