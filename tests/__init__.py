"""
LegalQA Test Suite

Comprehensive testing framework for performance optimizations and
functionality validation.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_CONFIG: Dict[str, Any] = {
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
        "max_cache_memory_mb": 512,  # MB
        "max_cache_access_time": 0.002,  # seconds
        "max_qa_latency": 5.0,  # seconds
    },
    "test_database": {"name": "legalqa_test", "sample_size": 100},
}

# E501 fix: long line wrapping
LONG_STRING = (
    "This is a very long string that needs to be wrapped across multiple lines "
    "to avoid exceeding the 79 character limit."
)
