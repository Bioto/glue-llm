# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in GlueLLM, please report it responsibly.

**Please do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please email security concerns to: [security@example.com] (replace with actual email)

Include the following information:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will:
- Acknowledge receipt within 48 hours
- Provide an initial assessment within 7 days
- Keep you informed of progress
- Credit you in the security advisory (if desired)

## Security Best Practices

When using GlueLLM:

1. **API Keys**: Never commit API keys to version control
   - Use environment variables or `.env` files (not committed)
   - Rotate keys regularly
   - Use least-privilege access when possible

2. **Tool Functions**: Be careful with tool implementations
   - Validate all inputs
   - Sanitize outputs when necessary
   - Don't execute arbitrary code from LLM responses

3. **Conversation Data**: Be mindful of sensitive data
   - Don't log conversations containing PII
   - Consider data retention policies
   - Encrypt stored conversation history

4. **Dependencies**: Keep dependencies updated
   - Regularly update `any-llm-sdk` and other dependencies
   - Review dependency security advisories

5. **Network**: Use secure connections
   - Always use HTTPS for API calls
   - Verify SSL certificates
   - Consider using VPNs for sensitive operations

## Known Security Considerations

- **Tool Execution**: Tools execute with the same permissions as your Python process
- **API Key Storage**: Keys are read from environment variables (use secure key management in production)
- **Conversation History**: Stored in memory by default (consider encryption for persistent storage)

## Disclosure Policy

We follow responsible disclosure practices:
- Vulnerabilities will be patched before public disclosure
- Security updates will be released promptly
- Users will be notified through GitHub releases and security advisories

Thank you for helping keep GlueLLM secure!
