"""
LegalQA Test Suite

Comprehensive testing framework for performance optimizations and functionality validation.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_CONFIG = {
    "sample_questions": [
        "Mi a bűnszervezet fogalma a Btk. szerint?",
        "Milyen jogkövetkezmények vonatkoznak a korrupciós bűncselekményekre?",
        "Hogyan kell értelmezni az önvédelem fogalmát?",
        "Mi a különbség a csalás és a sikkasztás között?",
        "Milyen feltételek mellett alkalmazható a feltételes szabadságra bocsátás?",
    ],
    "performance_thresholds": {
        "max_response_time": 2.0,  # seconds
        "max_startup_time": 10.0,  # seconds
        "max_memory_mb": 1024,  # MB
        "min_cache_hit_rate": 0.3,  # 30%
    },
    "test_database": {"name": "legalqa_test", "sample_size": 100},
}

# E501 javítás: hosszú sorok tördelése
LONG_STRING = (
    "Ez egy nagyon hosszú string, amit több sorba kell tördelni, hogy ne lépje túl a 79 karaktert."
)
