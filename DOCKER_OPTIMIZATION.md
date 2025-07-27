# Docker Optimizations

This document describes the Docker optimizations for the LegalQA project to speed up CI/CD processes.

## Dockerfile Variants

### 1. Dockerfile (Original)
- **Purpose**: Complete development environment
- **Dependencies**: All required packages
- **Build Time**: ~150 seconds
- **Usage**: Local development, production

### 2. Dockerfile.ci (CI Optimized)
- **Purpose**: CI/CD environment
- **Dependencies**: Full dependencies with better cache strategy
- **Build Time**: ~110 seconds
- **Usage**: GitHub Actions CI

### 3. Dockerfile.minimal (Minimal)
- **Purpose**: Fast CI testing
- **Dependencies**: Only essential packages
- **Build Time**: ~50 seconds
- **Usage**: GitHub Actions CI (current)

## Optimization Strategies

### 1. Layered Caching
- Installing dependencies in separate layers
- Smaller packages first
- Larger packages last

### 2. .dockerignore Optimization
- Excluding unnecessary files
- Reducing build context size

### 3. Multi-stage Build
- Separating builder and production stages
- Copying only necessary files

### 4. GitHub Actions Cache
- Using registry cache
- Optimizing layer cache

## Usage

### Local Development
```bash
docker build -t legalqa:dev .
```

### CI Testing
```bash
docker build -f Dockerfile.minimal -t legalqa:ci .
```

### Production Build
```bash
docker build -t legalqa:prod .
```

## Performance Comparison

| Dockerfile | Build Time | Size | Usage |
|------------|------------|------|-------|
| Original | ~150s | Large | Dev/Prod |
| CI | ~110s | Medium | CI |
| Minimal | ~50s | Small | CI |

## Future Optimizations

1. **Alpine Linux**: Smaller base image
2. **Pre-built Wheels**: Using pre-compiled packages
3. **Distroless**: Minimal runtime image
4. **Build Cache**: Better cache strategies

## Troubleshooting

### Build Timeout
- Use `Dockerfile.minimal` for CI
- Increase timeout in workflow
- Check cache settings

### Cache Issues
- Clear old caches
- Update cache keys
- Use registry cache
- If GitHub Actions cache fails, use fallback without cache
- Check GitHub service status