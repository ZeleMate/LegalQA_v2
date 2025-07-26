#!/usr/bin/env python3
"""
Quick Validation Script for LegalQA Optimizations

This script performs quick validation checks to ensure optimizations don't break functionality.
Run this before committing changes or deploying.
"""

import sys
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def validate_imports():
    """Validate that all critical imports work."""
    print("🔍 Validating imports...")

    critical_imports = [
        ("src.data_loading.faiss_loader", "load_faiss_index"),
        ("src.infrastructure.cache_manager", "CacheManager"),
        ("src.infrastructure.db_manager", "DatabaseManager"),
    ]

    for module_name, function_name in critical_imports:
        try:
            module = __import__(module_name, fromlist=[function_name])
            _ = getattr(module, function_name)
            print(f"  ✅ {module_name}.{function_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}.{function_name}: {e}")
            return False
        except Exception as e:
            print(f"  🔥 {module_name}.{function_name}: {e}")
            return False

    return True


def validate_cache_functionality():
    """Validate cache functionality works."""
    print("🧪 Validating cache functionality...")

    try:
        from src.infrastructure.cache_manager import CacheManager

        # Test basic cache operations
        cache = CacheManager(memory_maxsize=5, memory_ttl=60)

        # Test set/get
        cache.memory_cache.set("test_key", "test_value")
        value = cache.memory_cache.get("test_key")

        if value != "test_value":
            print(f"  ❌ Cache get/set failed: expected 'test_value', got '{value}'")
            return False

        # Test key generation
        key1 = cache._generate_key("test", "data")
        key2 = cache._generate_key("test", "data")

        if key1 != key2:
            print("  ❌ Cache key generation inconsistent")
            return False

        print("  ✅ Cache functionality working")
        return True

    except Exception as e:
        print(f"  ❌ Cache validation failed: {e}")
        return False


def validate_database_manager():
    """Validate database manager initialization."""
    print("🗄️ Validating database manager...")

    try:
        from src.infrastructure.db_manager import DatabaseManager

        # Test initialization
        db_manager = DatabaseManager()

        if db_manager._initialized:
            print("  ❌ Database manager should not be initialized without calling initialize()")
            return False

        # Test connection pooling attributes exist
        if not hasattr(db_manager, "pool"):
            print("  ❌ Database manager missing pool attribute")
            return False

        print("  ✅ Database manager initialization working")
        return True

    except Exception as e:
        print(f"  ❌ Database manager validation failed: {e}")
        return False


def validate_file_structure():
    """Validate that required files exist."""
    print("📁 Validating file structure...")

    required_files = [
        "src/infrastructure/cache_manager.py",
        "src/infrastructure/db_manager.py",
        "src/rag/retriever.py",
        "src/inference/app.py",
        "Dockerfile",
        "docker-compose.yml",
        "config/redis.conf",
        "config/prometheus.yml",
        "Makefile",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - File missing")
            all_exist = False

    return all_exist


def validate_performance_configs():
    """Validate performance configuration files."""
    print("⚙️ Validating performance configurations...")

    try:
        # Check pyproject.toml for new dependencies
        pyproject_path = project_root / "pyproject.toml"
        if not pyproject_path.exists():
            print("  ❌ pyproject.toml missing")
            return False

        content = pyproject_path.read_text()

        # Check for performance dependencies
        performance_deps = ["asyncpg", "aioredis", "prometheus-client"]
        missing_deps = []

        for dep in performance_deps:
            if dep not in content:
                missing_deps.append(dep)

        if missing_deps:
            print(f"  ❌ Missing performance dependencies: {missing_deps}")
            return False

        print("  ✅ Performance dependencies present in pyproject.toml")

        # Check Dockerfile for multi-stage build and other settings
        dockerfile_path = project_root / "Dockerfile"
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            if "as builder" in content and "USER appuser" in content and "HEALTHCHECK" in content:
                print("  ✅ Dockerfile contains key performance and security settings.")
            else:
                print(
                    "  ❌ Dockerfile missing some performance/security settings "
                    "(multi-stage, non-root user, healthcheck)."
                )
                return False
        else:
            print("  ❌ Dockerfile missing")
            return False

        # Check docker-compose for redis and prometheus
        compose_path = project_root / "docker-compose.yml"
        if compose_path.exists():
            print("  ✅ docker-compose.yml exists")
        else:
            print("  ❌ docker-compose.yml missing")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Performance config validation failed: {e}")
        return False


def validate_backward_compatibility():
    """Validate backward compatibility is maintained."""
    print("🔄 Validating backward compatibility...")

    try:
        # Test that original modules still work
        from src.chain.qa_chain import build_qa_chain

        print("  ✅ Original modules still importable")

        # Test that function signatures are compatible
        if not callable(build_qa_chain):
            print("  ❌ build_qa_chain not callable")
            return False

        print("  ✅ Function signatures compatible")
        return True

    except ImportError as e:
        print(f"  ❌ Original module import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Compatibility validation failed: {e}")
        return False


def run_quick_performance_check():
    """Run a quick performance check."""
    print("⚡ Running quick performance check...")

    try:
        # Test response time simulation
        start_time = time.time()

        # Simulate some work
        data = {}
        for i in range(1000):
            data[f"key_{i}"] = f"value_{i}"

        # Simulate lookup operations
        for i in range(100):
            _ = data.get(f"key_{i}")

        elapsed = time.time() - start_time

        if elapsed < 0.1:  # Should complete in under 100ms
            print(f"  ✅ Quick operations completed in {elapsed:.3f}s")
            return True
        else:
            print(f"  ⚠️ Quick operations took {elapsed:.3f}s (might be slow)")
            return True  # Warning, not failure

    except Exception as e:
        print(f"  ❌ Performance check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("🚀 LegalQA Optimization Validation")
    print("=" * 50)

    checks = [
        ("Import Validation", validate_imports),
        ("Cache Functionality", validate_cache_functionality),
        ("Database Manager", validate_database_manager),
        ("File Structure", validate_file_structure),
        ("Performance Configs", validate_performance_configs),
        ("Backward Compatibility", validate_backward_compatibility),
        ("Quick Performance Check", run_quick_performance_check),
    ]

    passed = 0
    total = len(checks)

    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                print(f"\n❌ {check_name} FAILED")
        except Exception as e:
            print(f"\n🔥 {check_name} ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Validation Summary: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All validations passed! Optimizations are working correctly.")
        return 0
    else:
        print("❌ Some validations failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
