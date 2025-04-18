# DataDog Logging Setup with FluentBit

## Prerequisites

1. Install FluentBit on your system
2. Obtain a DataDog API key

## Configuration

1. Set your DataDog API key as an environment variable:

```bash
export DD_API_KEY=your_datadog_api_key_here
export DD_ENV=development  # or production, staging, etc.
```

2. Start FluentBit with the provided configuration:

```bash
fluent-bit -c fluent-bit.conf
```

## Verifying Logs in DataDog

1. Log into your DataDog account
2. Navigate to Logs → Search
3. Filter by service:intelligent-chat-service to see your application logs

## Integration with Kubernetes

If deploying this application to Kubernetes, add the following annotations to your pod spec:

```yaml
annotations:
  ad.datadoghq.com/intelligent-chat-service.logs: '[{"source": "python", "service": "intelligent-chat-service"}]'
```

This will instruct the DataDog agent to collect logs from this service.
