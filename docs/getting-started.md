# Getting Started with AlphaMind

## Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher
- Git
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abrar2030/AlphaMind.git
cd AlphaMind
```

2. Set up the Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Install frontend dependencies:
```bash
cd ../frontend
npm install
```

## Configuration

1. Backend Configuration:
   - Copy `.env.example` to `.env` in the backend directory
   - Update the environment variables as needed

2. Frontend Configuration:
   - Copy `.env.example` to `.env` in the frontend directory
   - Update the environment variables as needed

## Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

The application should now be accessible at `http://localhost:3000`

## Testing

Run the test suite:
```bash
./test_components.sh
```

## Additional Resources

- [Architecture Overview](architecture.md)
- [API Documentation](api-documentation.md)
- [Development Guide](development-guide.md)
- [Deployment Guide](deployment.md)
- [Troubleshooting](troubleshooting.md)
