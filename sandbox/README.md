# Code Execution Sandbox

A secure sandboxed environment for executing Python code snippets, built with FastAPI and Docker.

## Overview

This service provides a secure way to execute untrusted Python code within an isolated Docker container. It exposes a REST API that allows clients to:

1. Execute Python code snippets
2. Upload and download data
3. Access common data science libraries like Pandas, NumPy, Matplotlib, etc.

## Getting Started

### Prerequisites

- Docker
- Docker Compose (optional)

### Building and Running

1. Build the Docker image:

```bash
docker build -t code-sandbox .
```

2. Run the container:

```bash
docker run -p 8000:8000 code-sandbox
```

3. If the main API server is running on 8000 use 80001
```bash
# Build the sandbox image 
docker build -t code-sandbox .

# Run the sandbox container on port 8001
docker run -d -p 8001:8001 --name code-sandbox code-sandbox
```

The API will be available at `http://localhost:8000`.
or The API will be available at `http://localhost:8001`.

## API Endpoints

### Health Check

```
GET /health
```

Returns the health status of the service.

### Execute Code

```
POST /execute_code
```

Executes Python code and returns the results.

Request body:
```json
{
  "code": "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})\ndf['c'] = df['a'] + df['b']\nprint(df)",
  "input_data": {}
}
```

Response:
```json
{
  "result": {
    "df": [
      {"a": 1, "b": 4, "c": 5},
      {"a": 2, "b": 5, "c": 7},
      {"a": 3, "b": 6, "c": 9}
    ]
  },
  "stdout": "   a  b  c\n0  1  4  5\n1  2  5  7\n2  3  6  9\n",
  "stderr": "",
  "execution_time": 0.0234
}
```

### Upload Data

```
POST /upload_data/{key}
```

Uploads data to the sandbox for later use.

### Get Data

```
GET /get_data/{key}
```

Retrieves previously uploaded data.

### Clear Data

```
GET /clear_data
```

Clears all stored data.

## Security Considerations

This sandbox:

1. Runs in an isolated Docker container with limited resources
2. Only provides access to a safe subset of Python libraries
3. Doesn't allow file system access outside its container
4. Has execution time limits to prevent long-running operations

## Note on Usage

This sandbox is designed for executing data science and analytics code in a controlled environment. It is NOT designed for:

1. Web scraping or external network access
2. Security-critical applications
3. File system operations
4. Long-running processes

## License

MIT 