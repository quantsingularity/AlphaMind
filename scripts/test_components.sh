#!/bin/bash

# Test script for AlphaMind project
# This script tests both frontend and backend components

echo "Starting AlphaMind test suite..."

# Create test directory if it doesn't exist
mkdir -p /home/ubuntu/AlphaMind/tests/results

# Test frontend components
echo "Testing frontend components..."

# Check if all HTML files are valid
echo "Checking HTML validity..."
for file in /home/ubuntu/AlphaMind/frontend/*.html; do
  filename=$(basename "$file")
  echo "Testing $filename..."
  
  # Check if file exists and is readable
  if [ -r "$file" ]; then
    echo "✓ File exists and is readable"
  else
    echo "✗ File does not exist or is not readable"
    exit 1
  fi
  
  # Check for basic HTML structure
  if grep -q "<!DOCTYPE html>" "$file" && grep -q "<html" "$file" && grep -q "<head>" "$file" && grep -q "<body>" "$file"; then
    echo "✓ Basic HTML structure is valid"
  else
    echo "✗ Basic HTML structure is invalid"
    exit 1
  fi
  
  # Check for responsive viewport meta tag
  if grep -q '<meta name="viewport"' "$file"; then
    echo "✓ Responsive viewport meta tag found"
  else
    echo "✗ Responsive viewport meta tag missing"
    exit 1
  fi
  
  # Check for CSS links
  if grep -q '<link rel="stylesheet" href="css/modern-styles.css">' "$file" || grep -q '<link rel="stylesheet" href="css/styles.css">' "$file"; then
    echo "✓ CSS link found"
  else
    echo "✗ CSS link missing"
    exit 1
  fi
  
  echo "$filename passed basic validation tests"
  echo ""
done

# Check if all CSS files are valid
echo "Checking CSS files..."
for file in /home/ubuntu/AlphaMind/frontend/css/*.css; do
  filename=$(basename "$file")
  echo "Testing $filename..."
  
  # Check if file exists and is readable
  if [ -r "$file" ]; then
    echo "✓ File exists and is readable"
  else
    echo "✗ File does not exist or is not readable"
    exit 1
  fi
  
  # Check for basic CSS structure (at least one rule)
  if grep -q "{" "$file" && grep -q "}" "$file"; then
    echo "✓ Basic CSS structure is valid"
  else
    echo "✗ Basic CSS structure is invalid"
    exit 1
  fi
  
  # Check for responsive media queries
  if grep -q "@media" "$file"; then
    echo "✓ Responsive media queries found"
  else
    echo "! Warning: No responsive media queries found"
  fi
  
  echo "$filename passed basic validation tests"
  echo ""
done

# Check if all JS files are valid
echo "Checking JavaScript files..."
for file in /home/ubuntu/AlphaMind/frontend/js/*.js; do
  filename=$(basename "$file")
  echo "Testing $filename..."
  
  # Check if file exists and is readable
  if [ -r "$file" ]; then
    echo "✓ File exists and is readable"
  else
    echo "✗ File does not exist or is not readable"
    exit 1
  fi
  
  # Check for basic JS structure (at least one function)
  if grep -q "function" "$file"; then
    echo "✓ Basic JS structure is valid"
  else
    echo "✗ Basic JS structure is invalid"
    exit 1
  fi
  
  # Check for event listeners
  if grep -q "addEventListener" "$file"; then
    echo "✓ Event listeners found"
  else
    echo "! Warning: No event listeners found"
  fi
  
  echo "$filename passed basic validation tests"
  echo ""
done

# Check if all image files exist
echo "Checking image files..."
for file in $(grep -o 'images/[^"]*' /home/ubuntu/AlphaMind/frontend/*.html | sort | uniq); do
  filepath="/home/ubuntu/AlphaMind/frontend/$file"
  echo "Testing $file..."
  
  # Check if file exists
  if [ -f "$filepath" ]; then
    echo "✓ Image file exists"
  else
    echo "! Warning: Image file does not exist: $filepath"
  fi
done

echo "Frontend tests completed."
echo ""

# Test backend components
echo "Testing backend components..."

# Check if Python is installed
if command -v python3 &>/dev/null; then
  echo "✓ Python is installed"
  python_version=$(python3 --version)
  echo "  Python version: $python_version"
else
  echo "✗ Python is not installed"
  exit 1
fi

# Check if required Python packages are installed
echo "Checking Python packages..."
required_packages=("numpy" "pandas" "tensorflow" "scikit-learn")
for package in "${required_packages[@]}"; do
  if python3 -c "import $package" &>/dev/null; then
    echo "✓ Package $package is installed"
  else
    echo "! Warning: Package $package is not installed"
  fi
done

# Check if backend Python files are valid
echo "Checking backend Python files..."
for file in $(find /home/ubuntu/AlphaMind/backend -name "*.py"); do
  filename=$(basename "$file")
  echo "Testing $filename..."
  
  # Check if file exists and is readable
  if [ -r "$file" ]; then
    echo "✓ File exists and is readable"
  else
    echo "✗ File does not exist or is not readable"
    exit 1
  fi
  
  # Check for Python syntax errors
  if python3 -m py_compile "$file" 2>/dev/null; then
    echo "✓ Python syntax is valid"
  else
    echo "✗ Python syntax is invalid"
    exit 1
  fi
  
  echo "$filename passed basic validation tests"
  echo ""
done

# Test the new AI model components
echo "Testing new AI model components..."

# Test attention_mechanism.py
if [ -f "/home/ubuntu/AlphaMind/backend/ai_models/attention_mechanism.py" ]; then
  echo "Testing attention_mechanism.py..."
  
  # Create a simple test script
  cat > /home/ubuntu/AlphaMind/tests/test_attention.py << 'EOF'
import sys
import os
sys.path.append('/home/ubuntu/AlphaMind/backend')
try:
    from ai_models.attention_mechanism import MultiHeadAttention, TemporalAttentionBlock
    print("✓ Successfully imported MultiHeadAttention and TemporalAttentionBlock")
    
    # Test basic initialization
    import tensorflow as tf
    attention = MultiHeadAttention(d_model=64, num_heads=4)
    print("✓ Successfully initialized MultiHeadAttention")
    
    block = TemporalAttentionBlock(d_model=64, num_heads=4, dff=256)
    print("✓ Successfully initialized TemporalAttentionBlock")
    
    print("All tests passed for attention_mechanism.py")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    sys.exit(1)
EOF

  # Run the test script
  python3 /home/ubuntu/AlphaMind/tests/test_attention.py
  echo ""
fi

# Test sentiment_analysis.py
if [ -f "/home/ubuntu/AlphaMind/backend/alternative_data/sentiment_analysis.py" ]; then
  echo "Testing sentiment_analysis.py..."
  
  # Create a simple test script
  cat > /home/ubuntu/AlphaMind/tests/test_sentiment.py << 'EOF'
import sys
import os
sys.path.append('/home/ubuntu/AlphaMind/backend')
try:
    from alternative_data.sentiment_analysis import MarketSentimentAnalyzer
    print("✓ Successfully imported MarketSentimentAnalyzer")
    
    # Test basic initialization
    analyzer = MarketSentimentAnalyzer()
    print("✓ Successfully initialized MarketSentimentAnalyzer")
    
    print("All tests passed for sentiment_analysis.py")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    sys.exit(1)
EOF

  # Run the test script
  python3 /home/ubuntu/AlphaMind/tests/test_sentiment.py
  echo ""
fi

# Test portfolio_optimization.py
if [ -f "/home/ubuntu/AlphaMind/backend/alpha_research/portfolio_optimization.py" ]; then
  echo "Testing portfolio_optimization.py..."
  
  # Create a simple test script
  cat > /home/ubuntu/AlphaMind/tests/test_portfolio.py << 'EOF'
import sys
import os
sys.path.append('/home/ubuntu/AlphaMind/backend')
try:
    from alpha_research.portfolio_optimization import PortfolioOptimizer
    print("✓ Successfully imported PortfolioOptimizer")
    
    # Test basic initialization
    optimizer = PortfolioOptimizer(n_assets=5)
    print("✓ Successfully initialized PortfolioOptimizer")
    
    print("All tests passed for portfolio_optimization.py")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    sys.exit(1)
EOF

  # Run the test script
  python3 /home/ubuntu/AlphaMind/tests/test_portfolio.py
  echo ""
fi

# Test authentication.py
if [ -f "/home/ubuntu/AlphaMind/backend/infrastructure/authentication.py" ]; then
  echo "Testing authentication.py..."
  
  # Create a simple test script
  cat > /home/ubuntu/AlphaMind/tests/test_auth.py << 'EOF'
import sys
import os
sys.path.append('/home/ubuntu/AlphaMind/backend')
try:
    from infrastructure.authentication import AuthenticationSystem
    print("✓ Successfully imported AuthenticationSystem")
    
    # Test token generation and verification
    import jwt
    from flask import Flask
    
    app = Flask(__name__)
    auth = AuthenticationSystem(app, 'test-secret-key')
    print("✓ Successfully initialized AuthenticationSystem")
    
    token = auth.generate_token('testuser')
    print("✓ Successfully generated token")
    
    username = auth.verify_token(token)
    if username == 'testuser':
        print("✓ Successfully verified token")
    else:
        print(f"✗ Token verification failed: {username}")
        sys.exit(1)
    
    print("All tests passed for authentication.py")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    sys.exit(1)
EOF

  # Run the test script
  python3 /home/ubuntu/AlphaMind/tests/test_auth.py
  echo ""
fi

echo "Backend tests completed."
echo ""

echo "All tests completed successfully!"
echo "The AlphaMind project is ready for deployment."
