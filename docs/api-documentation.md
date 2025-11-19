# API Documentation

## Base URL
```
https://api.alphamind.com/v1
```

## Authentication

All API requests require authentication using JWT tokens.

### Authentication Headers
```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### User Management

#### Register User
```
POST /users/register
```
Request Body:
```json
{
    "username": "string",
    "email": "string",
    "password": "string"
}
```

#### Login
```
POST /users/login
```
Request Body:
```json
{
    "email": "string",
    "password": "string"
}
```

#### Get User Profile
```
GET /users/profile
```

### AI Features

#### Generate Response
```
POST /ai/generate
```
Request Body:
```json
{
    "prompt": "string",
    "max_tokens": "integer",
    "temperature": "float"
}
```

#### Get AI History
```
GET /ai/history
```

### System Management

#### Health Check
```
GET /health
```

#### System Status
```
GET /status
```

## Response Format

All responses follow this format:
```json
{
    "status": "success|error",
    "data": {},
    "message": "string",
    "timestamp": "ISO8601"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 500 | Internal Server Error |

## Rate Limiting

- 100 requests per minute per IP
- 1000 requests per hour per user

## Pagination

Endpoints that return lists support pagination:

```
GET /endpoint?page=1&limit=10
```

## WebSocket API

### Connection
```
wss://api.alphamind.com/v1/ws
```

### Events
- `message`: Real-time message updates
- `status`: System status updates
- `error`: Error notifications

## SDKs

Official SDKs are available for:
- Python
- JavaScript/TypeScript
- Java
- Go

## Examples

### Python
```python
import requests

headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

response = requests.post(
    'https://api.alphamind.com/v1/ai/generate',
    headers=headers,
    json={
        'prompt': 'Hello, world!',
        'max_tokens': 100
    }
)
```

### JavaScript
```javascript
fetch('https://api.alphamind.com/v1/ai/generate', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        prompt: 'Hello, world!',
        max_tokens: 100
    })
});
```
