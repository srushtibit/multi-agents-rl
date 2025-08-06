"""Version information for the Multilingual Multi-Agent Support System."""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

# Version metadata
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_BUILD = "stable"

# Build information
BUILD_DATE = "2025-01-07"
BUILD_COMMIT = "main"

# Feature flags for this version
FEATURES = {
    "multilingual_support": True,
    "reinforcement_learning": True,
    "symbolic_communication": True,
    "escalation_system": True,
    "knowledge_base": True,
    "streamlit_ui": True,
    "api_endpoints": True,
    "email_notifications": True,
    "document_processing": True,
    "semantic_search": True,
}

def get_version_string() -> str:
    """Get formatted version string."""
    if VERSION_BUILD and VERSION_BUILD != "stable":
        return f"{__version__}-{VERSION_BUILD}"
    return __version__

def get_version_info() -> dict:
    """Get comprehensive version information."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build_date": BUILD_DATE,
        "build_commit": BUILD_COMMIT,
        "features": FEATURES,
    }