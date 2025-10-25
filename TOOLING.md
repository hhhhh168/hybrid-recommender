# Modern Tooling Guide

This project uses three modern development tools to streamline workflows, improve dependency management, and standardize commits.

## Table of Contents
- [Just - Command Runner](#just---command-runner)
- [UV - Python Package Manager](#uv---python-package-manager)
- [Commitizen - Conventional Commits](#commitizen---conventional-commits)
- [Quick Start](#quick-start)
- [Common Workflows](#common-workflows)

---

## Just - Command Runner

**Just** is a modern command runner that simplifies project tasks. Think of it as a developer-friendly alternative to Make.

### Installation

#### macOS
```bash
brew install just
```

#### Linux
```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

#### Windows
```bash
cargo install just
# or
scoop install just
```

### Usage

View all available commands:
```bash
just
```

Common commands:
```bash
just setup           # Create virtual environment
just install         # Install dependencies
just test            # Run tests
just eval-quick      # Quick evaluation (1k users)
just eval-full       # Full evaluation (20k users)
just clean           # Clean generated files
just status          # Show project status
just dev-cycle       # Full dev cycle: clean → generate → eval
```

### Custom Parameters

Some recipes accept parameters:
```bash
just eval 5 10 20 0.2 1000  # Custom K values, test size, max users
```

### Learn More
- Official site: https://just.systems
- GitHub: https://github.com/casey/just

---

## UV - Python Package Manager

**UV** is an extremely fast Python package manager (10-100x faster than pip) built by Astral, the creators of Ruff.

### Installation

#### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Via pip
```bash
pip install uv
```

### Usage

#### Create a virtual environment
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

#### Install dependencies
```bash
uv pip install -r requirements.txt
```

#### Install from pyproject.toml
```bash
uv pip install -e .
uv pip install -e ".[dev]"  # Include dev dependencies
```

#### Add a new package
```bash
uv pip install numpy
uv pip freeze > requirements.txt  # Update requirements
```

#### Install specific Python version
```bash
uv python install 3.11
uv python pin 3.11  # Updates .python-version
```

### Speed Comparison

| Tool | Time to install pandas |
|------|------------------------|
| pip  | ~4.0s                  |
| UV   | ~0.06s                 |

That's **67x faster**! 🚀

### Learn More
- Official docs: https://docs.astral.sh/uv/
- GitHub: https://github.com/astral-sh/uv

---

## Commitizen - Conventional Commits

**Commitizen** enforces conventional commit messages, automates version bumping, and generates changelogs.

### Installation

Already included in dev dependencies:
```bash
uv pip install -e ".[dev]"
# or
just install-dev
```

### Usage

#### Create a conventional commit
Instead of `git commit -m "message"`, use:
```bash
cz commit
# or via Just
just commit
```

This will prompt you to select commit type and provide details:
```
? Select the type of change you are committing: (Use arrow keys)
 » feat: A new feature
   fix: A bug fix
   docs: Documentation only changes
   style: Changes that do not affect the meaning of the code
   refactor: A code change that neither fixes a bug nor adds a feature
   perf: A performance improvement
   test: Adding missing tests
```

#### Bump version and generate changelog
```bash
cz bump --changelog
# or via Just
just bump
```

This will:
1. Analyze commits since last version
2. Determine version bump (major/minor/patch)
3. Update version in `pyproject.toml`
4. Generate/update `CHANGELOG.md`
5. Create a git tag

### Commit Message Format

Commitizen enforces this format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Examples:**
```
feat: add user preference filtering to NLP model
fix: correct NDCG calculation for ties
docs: update README with evaluation metrics
perf: optimize similarity matrix computation
refactor: extract data loading into utils module
test: add unit tests for collaborative filtering
```

### Conventional Commit Types

| Type       | Description                                      | Version Bump |
|------------|--------------------------------------------------|--------------|
| `feat`     | New feature                                      | MINOR        |
| `fix`      | Bug fix                                          | PATCH        |
| `perf`     | Performance improvement                          | PATCH        |
| `docs`     | Documentation only                               | None         |
| `style`    | Code style changes (formatting, whitespace)      | None         |
| `refactor` | Code refactoring (no functional changes)         | PATCH*       |
| `test`     | Adding/updating tests                            | None         |
| `chore`    | Build process, dependencies, tooling             | None         |

*Configured to bump patch version in this project

### Learn More
- Official docs: https://commitizen-tools.github.io/commitizen/
- Conventional Commits: https://www.conventionalcommits.org/
- GitHub: https://github.com/commitizen-tools/commitizen

---

## Quick Start

### First Time Setup

1. **Install the tools:**
   ```bash
   # macOS
   brew install just
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Linux
   curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Setup the project:**
   ```bash
   just setup        # Create venv
   source .venv/bin/activate
   just install-dev  # Install all dependencies including dev tools
   ```

3. **Verify installation:**
   ```bash
   just --version
   uv --version
   cz version
   ```

### Daily Development

```bash
# Activate virtual environment
source .venv/bin/activate

# View available commands
just

# Run quick development cycle
just dev-cycle

# Make changes, then commit using conventional commits
git add .
just commit

# Run tests
just test
```

---

## Common Workflows

### Adding a New Feature

```bash
# 1. Create a branch
git checkout -b feature/new-algorithm

# 2. Make changes to code

# 3. Add dependencies if needed
uv pip install new-package
uv pip freeze > requirements.txt

# 4. Run tests
just test

# 5. Commit with conventional format
git add .
just commit
# Select: feat
# Description: "add new ranking algorithm"

# 6. Push and create PR
git push origin feature/new-algorithm
```

### Running Experiments

```bash
# Clean everything
just clean

# Generate fresh data
just generate-data

# Run quick evaluation
just eval-quick

# Or run custom evaluation
just eval "3 5 10" 0.3 500

# Check results
ls -lh results/metrics/
```

### Releasing a New Version

```bash
# 1. Ensure all changes are committed
git status

# 2. Bump version and generate changelog
just bump

# This will:
# - Analyze commits since last tag
# - Determine version bump (0.1.0 → 0.2.0 for feat, → 0.1.1 for fix)
# - Update pyproject.toml
# - Generate/update CHANGELOG.md
# - Create git tag v0.2.0

# 3. Push changes and tags
git push && git push --tags
```

### Checking Project Health

```bash
# View project status
just status

# Run full check (lint + test)
just check

# Clean and reset everything
just reset
```

---

## Integration Benefits

### Speed
- **UV**: 10-100x faster dependency installation
- **Just**: Instant command execution (no Makefile parsing)

### Developer Experience
- **Just**: Simple, readable command definitions
- **UV**: Unified tool for all Python needs
- **Commitizen**: Interactive commit prompts

### Automation
- **Commitizen**: Automatic version bumping and changelog generation
- **Just**: Composite commands for complete workflows
- **UV**: Reproducible environments with lock files

### Standards
- **Commitizen**: Enforced conventional commits
- **Just**: Documented, discoverable commands
- **UV**: Modern Python packaging standards

---

## Troubleshooting

### Just command not found
```bash
# Add to PATH (Linux/macOS)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### UV not activating virtual environment
```bash
# Manually activate
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Commitizen errors
```bash
# Ensure commitizen is installed
uv pip install commitizen

# Check configuration
cat pyproject.toml | grep -A 10 "tool.commitizen"
```

### Just recipe fails
```bash
# Run with verbose output
just --verbose <recipe-name>

# Check if you're in project root
pwd  # Should show .../hybrid-recommender
```

---

## Additional Resources

- **Just Cheatsheet**: https://cheatography.com/linux-china/cheat-sheets/justfile/
- **UV Guide**: https://www.datacamp.com/tutorial/python-uv
- **Commitizen Tutorial**: https://commitizen-tools.github.io/commitizen/tutorials/writing_commits/
- **Conventional Commits**: https://www.conventionalcommits.org/en/v1.0.0/

---

**Last Updated:** 2025-10-25
