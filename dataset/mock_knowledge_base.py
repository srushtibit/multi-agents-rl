
# dataset/mock_knowledge_base.py

mock_kb = {
    "ticket_1001": {
        "issue": "Email not syncing",
        "solution": "Check network connection, then reconfigure email client settings. If issue persists, escalate to IT Level 2.",
        "category": "IT Support"
    },
    "ticket_1002": {
        "issue": "Payroll discrepancy",
        "solution": "Verify timesheet entries against payroll records. If discrepancy found, submit a payroll adjustment request form to HR.",
        "category": "HR Support"
    },
    "ticket_1003": {
        "issue": "VPN connection failed",
        "solution": "Restart VPN client, verify credentials. Ensure no firewall is blocking the connection. Contact IT Helpdesk if problem continues.",
        "category": "IT Support"
    },
    "ticket_1004": {
        "issue": "Benefits enrollment question",
        "solution": "Refer to the NexaCorp HR Manual, Section 3.2 on Benefits Enrollment. Contact HR department for personalized assistance.",
        "category": "HR Support"
    },
    "email_sync_troubleshooting": {
        "issue": "General email sync issues",
        "solution": "Common steps include verifying internet connectivity, checking server status, updating email client, and ensuring correct port settings.",
        "category": "IT Support"
    },
    "payroll_adjustment_process": {
        "issue": "How to request payroll adjustment",
        "solution": "Fill out the \"Payroll Adjustment Request\" form available on the HR portal. Attach supporting documents like corrected timesheets. Submit to your manager for approval.",
        "category": "HR Support"
    }
}

def query_knowledge_base(query: str) -> dict:
    """
    Simulates querying a knowledge base.
    In a real system, this would involve semantic search, keyword matching, etc.
    For now, it does a simple keyword search across issues and solutions.
    """
    results = {}
    query_lower = query.lower()
    for key, value in mock_kb.items():
        if query_lower in value.get("issue", "").lower() or \
           query_lower in value.get("solution", "").lower() or \
           query_lower in key.lower():
            results[key] = value
    return results

