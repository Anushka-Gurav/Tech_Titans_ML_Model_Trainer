---
inclusion: always
---

# Agent Hooks Configuration Guide

This document provides pre-configured agent hooks for the ML Training Platform project to automate quality checks, environment validation, and development workflow consistency.

## Quick Setup Instructions

1. Open Command Palette in Kiro (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Search for "Open Kiro Hook UI"
3. Create new hooks using the configurations below
4. Copy the exact trigger, action, and content for each hook

## Hook Configurations

### 1. Environment Check Hook

**Purpose**: Verify development environment before starting tasks
**Trigger**: On message send (when starting a task)
**Action**: Send message to agent

```
Hook Name: Environment Check
Trigger: On message send
Action: Send message
Message: "🔍 ENVIRONMENT CHECK - Before starting this task, please verify:

✅ Python virtual environment is activated
✅ All dependencies from requirements.txt are installed
✅ Flask development server can start successfully
✅ React development server can start successfully
✅ All environment variables are properly configured
✅ Database connections (if applicable) are working

If any of these checks fail, address them before proceeding with the task.

Run these commands to verify:
- `python --version` (should show Python 3.9+)
- `pip list` (check installed packages)
- `npm --version` (should show Node.js 16+)
- `npm list` (check frontend dependencies)

Proceed only when environment is confirmed working."
```

### 2. Code Quality Validation Hook

**Purpose**: Run quality checks after completing implementation tasks
**Trigger**: On agent execution complete
**Action**: Execute command

```
Hook Name: Code Quality Check
Trigger: On agent execution complete
Action: Execute command
Command:
echo "🔍 Running Code Quality Checks..."

# Python Backend Checks
if [ -d "backend" ]; then
  echo "📋 Checking Python code quality..."
  cd backend

  echo "  → Running Black formatter check..."
  black --check --line-length 88 . || echo "❌ Black formatting needed: run 'black --line-length 88 .'"

  echo "  → Running Flake8 linting..."
  flake8 --max-line-length 88 --extend-ignore E203,W503 . || echo "❌ Flake8 issues found"

  echo "  → Running MyPy type checking..."
  mypy --strict . || echo "❌ MyPy type issues found"

  echo "  → Running pytest..."
  pytest --tb=short || echo "❌ Tests failing"

  cd ..
fi

# Frontend Checks
if [ -d "frontend" ]; then
  echo "📋 Checking React code quality..."
  cd frontend

  echo "  → Running ESLint..."
  npm run lint || echo "❌ ESLint issues found: run 'npm run lint:fix'"

  echo "  → Running TypeScript check..."
  npm run type-check || echo "❌ TypeScript issues found"

  echo "  → Running accessibility tests..."
  npm run test:a11y || echo "❌ Accessibility issues found"

  cd ..
fi

echo "✅ Code quality checks completed"
```

### 3. Accessibility Validation Hook

**Purpose**: Ensure accessibility standards when frontend files are modified
**Trigger**: On file save (_.tsx, _.jsx, _.ts, _.js files)
**Action**: Send message to agent

```
Hook Name: Accessibility Reminder
Trigger: On file save
File Pattern: *.tsx,*.jsx,*.ts,*.js
Action: Send message
Message: "♿ ACCESSIBILITY CHECK - Frontend component updated. Please ensure:

🏷️  **Labels & Forms**
   - All form inputs have proper labels (htmlFor attribute)
   - Required fields marked with aria-label='required'
   - Error messages use role='alert'

⌨️  **Keyboard Navigation**
   - All interactive elements are keyboard accessible
   - Focus indicators are visible and clear
   - Tab order is logical

🎨  **Visual Design**
   - Color contrast meets WCAG 2.1 AA standards (4.5:1 ratio)
   - Information not conveyed by color alone
   - Text is readable and scalable

🔊  **Screen Readers**
   - ARIA labels used for complex components
   - Live regions for dynamic content (aria-live)
   - Proper heading hierarchy (h1 → h2 → h3)

📊  **Data & Charts**
   - Tables have proper headers and captions
   - Charts include alternative text descriptions
   - Progress indicators have ARIA attributes

🧪  **Testing**
   Run: `npm run test:a11y` to validate accessibility compliance

Refer to the HTML Accessibility Guide for detailed examples."
```

### 4. Task Completion Validation Hook

**Purpose**: Validate task completion against requirements
**Trigger**: On agent execution complete
**Action**: Send message to agent

```
Hook Name: Task Completion Validation
Trigger: On agent execution complete
Action: Send message
Message: "✅ TASK COMPLETION CHECKLIST

📋 **Code Quality**
   - [ ] Code follows Python PEP 8 standards (backend)
   - [ ] React components follow accessibility guidelines
   - [ ] All functions have proper type hints and docstrings
   - [ ] Error handling is implemented correctly

🧪 **Testing**
   - [ ] Unit tests written and passing
   - [ ] Property-based tests implemented (if applicable)
   - [ ] Integration tests cover main workflows
   - [ ] Accessibility tests pass

📚 **Documentation**
   - [ ] Code is self-documenting with clear variable names
   - [ ] Complex logic has explanatory comments
   - [ ] API endpoints documented (if applicable)

🔗 **Requirements Traceability**
   - [ ] Task addresses specific requirements mentioned
   - [ ] All acceptance criteria are met
   - [ ] No scope creep beyond task definition

🚀 **Integration**
   - [ ] Code integrates properly with existing components
   - [ ] No breaking changes to other parts of system
   - [ ] Database migrations (if applicable) are included

Mark this task as complete only when all items are verified."
```

### 5. ML Model Training Hook

**Purpose**: Specific checks for ML training tasks
**Trigger**: Manual (button click)
**Action**: Execute command

```
Hook Name: ML Training Validation
Trigger: Manual
Action: Execute command
Command:
echo "🤖 ML Training Environment Check..."

# Check Python ML dependencies
echo "📦 Checking ML dependencies..."
python -c "
import sys
required_packages = ['scikit-learn', 'pandas', 'numpy', 'matplotlib']
missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package} installed')
    except ImportError:
        missing.append(package)
        print(f'❌ {package} missing')

if missing:
    print(f'Install missing packages: pip install {\" \".join(missing)}')
    sys.exit(1)
else:
    print('✅ All ML dependencies available')
"

# Check dataset directory
if [ -d "datasets" ]; then
    echo "✅ Dataset directory exists"
    echo "📊 Available datasets:"
    ls -la datasets/
else
    echo "❌ Dataset directory not found - create 'datasets' folder"
fi

# Check model storage
if [ -d "models" ]; then
    echo "✅ Model storage directory exists"
else
    echo "📁 Creating model storage directory..."
    mkdir -p models
fi

# Verify training configuration
if [ -f "backend/config/training_config.py" ]; then
    echo "✅ Training configuration found"
else
    echo "❌ Training configuration missing"
fi

echo "🎯 ML environment check completed"
```

### 6. Deployment Readiness Hook

**Purpose**: Check if application is ready for deployment
**Trigger**: Manual (button click)
**Action**: Execute command

```
Hook Name: Deployment Readiness
Trigger: Manual
Action: Execute command
Command:
echo "🚀 DEPLOYMENT READINESS CHECK"

# Security checks
echo "🔒 Security checks..."
echo "  → Checking for hardcoded secrets..."
grep -r "password\|secret\|key\|token" --include="*.py" --include="*.js" --include="*.ts" --exclude-dir=node_modules --exclude-dir=.git . | grep -v "# Safe:" || echo "✅ No obvious hardcoded secrets"

echo "  → Checking for debug flags..."
grep -r "DEBUG.*=.*True\|console\.log\|print(" --include="*.py" --include="*.js" --include="*.ts" --exclude-dir=node_modules --exclude-dir=.git . || echo "✅ No debug statements found"

# Environment checks
echo "🌍 Environment checks..."
echo "  → Checking for localhost references..."
grep -r "localhost\|127\.0\.0\.1" --exclude-dir=node_modules --exclude-dir=.git --exclude="*.md" . || echo "✅ No hardcoded localhost URLs"

echo "  → Checking environment variables..."
if [ -f ".env.example" ]; then
    echo "✅ Environment template found"
else
    echo "❌ Create .env.example with required variables"
fi

# Build tests
echo "🏗️  Build tests..."
if [ -d "backend" ]; then
    echo "  → Testing backend startup..."
    cd backend
    timeout 10s python -c "from app import app; print('✅ Backend imports successfully')" || echo "❌ Backend startup issues"
    cd ..
fi

if [ -d "frontend" ]; then
    echo "  → Testing frontend build..."
    cd frontend
    npm run build > /dev/null 2>&1 && echo "✅ Frontend builds successfully" || echo "❌ Frontend build issues"
    cd ..
fi

# Documentation check
echo "📚 Documentation check..."
[ -f "README.md" ] && echo "✅ README.md exists" || echo "❌ Create README.md"
[ -f "requirements.txt" ] && echo "✅ requirements.txt exists" || echo "❌ Create requirements.txt"
[ -f "package.json" ] && echo "✅ package.json exists" || echo "❌ Create package.json"

echo "🎯 Deployment readiness check completed"
```

### 7. Git Commit Quality Hook

**Purpose**: Ensure quality before commits
**Trigger**: Manual (before committing)
**Action**: Execute command

```
Hook Name: Pre-Commit Quality Check
Trigger: Manual
Action: Execute command
Command:
echo "📝 PRE-COMMIT QUALITY CHECK"

# Check git status
echo "📊 Git status:"
git status --porcelain

# Run linting on staged files
echo "🔍 Checking staged Python files..."
git diff --cached --name-only --diff-filter=ACM | grep "\.py$" | xargs -r black --check --line-length 88
git diff --cached --name-only --diff-filter=ACM | grep "\.py$" | xargs -r flake8 --max-line-length 88

echo "🔍 Checking staged JavaScript/TypeScript files..."
git diff --cached --name-only --diff-filter=ACM | grep -E "\.(js|jsx|ts|tsx)$" | xargs -r npx eslint

# Check commit message format
echo "💬 Commit message guidelines:"
echo "  Format: type(scope): description"
echo "  Types: feat, fix, docs, style, refactor, test, chore"
echo "  Example: feat(dataset): add CSV upload validation"

# Run tests on staged files
echo "🧪 Running tests..."
if git diff --cached --name-only | grep -q "\.py$"; then
    echo "  → Running Python tests..."
    pytest --tb=short
fi

if git diff --cached --name-only | grep -q -E "\.(js|jsx|ts|tsx)$"; then
    echo "  → Running frontend tests..."
    cd frontend && npm test -- --watchAll=false && cd ..
fi

echo "✅ Pre-commit checks completed"
echo "💡 If all checks pass, proceed with: git commit -m 'your message'"
```

## Hook Usage Guidelines

### When to Use Each Hook

1. **Environment Check** - Use at start of each development session
2. **Code Quality Check** - Automatically runs after task completion
3. **Accessibility Reminder** - Triggers when editing frontend components
4. **Task Completion Validation** - Use before marking tasks complete
5. **ML Training Validation** - Use before training models
6. **Deployment Readiness** - Use before deploying to staging/production
7. **Pre-Commit Quality** - Use before committing code to git

### Best Practices

- **Test hooks individually** after creating them
- **Modify commands** for your specific environment (Windows/Mac/Linux)
- **Add project-specific checks** as needed
- **Keep hook messages concise** but informative
- **Use emojis** for visual clarity in terminal output

### Troubleshooting

If hooks fail:

1. Check file paths are correct for your project structure
2. Ensure required tools are installed (black, flake8, mypy, etc.)
3. Verify shell commands work in your terminal
4. Adjust commands for your operating system

These hooks will significantly improve code quality and development workflow consistency for the ML Training Platform project.
