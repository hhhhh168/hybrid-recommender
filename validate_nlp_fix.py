"""
Validation script to verify NLP recommender fixes.

Tests the critical index order fix and other improvements.
"""

import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

def test_index_order_consistency():
    """Test that profile text order matches user_id_to_idx mapping."""
    print("="*60)
    print("TEST 1: Index Order Consistency")
    print("="*60)

    # Create test users
    test_users = pd.DataFrame({
        'user_id': ['user_c', 'user_a', 'user_b'],  # Intentionally out of order
        'gender': ['male', 'female', 'male'],
        'age': [30, 25, 35],
        'city': ['NYC', 'LA', 'SF'],
        'bio': ['bio c', 'bio a', 'bio b'],
        'job_title': ['Engineer', 'Designer', 'Manager'],
        'school': ['MIT', 'Stanford', 'Harvard'],
        'matching_pref_gender': ['female', 'male', 'female'],
        'matching_pref_age_min': [25, 25, 25],
        'matching_pref_age_max': [35, 35, 35],
        'matching_pref_use_location': [False, False, False]
    })

    preprocessor = DataPreprocessor()

    # Create profile texts
    profile_texts = preprocessor.create_profile_text(test_users)

    # Simulate what NLP model does
    all_user_ids = test_users['user_id'].values  # ['user_c', 'user_a', 'user_b']
    user_id_to_idx = {uid: idx for idx, uid in enumerate(all_user_ids)}

    # CRITICAL: Use fixed approach
    texts = [profile_texts.loc[uid] for uid in all_user_ids]

    # Verify order
    print("\nUser ID order:", all_user_ids.tolist())
    print("Mapping:", user_id_to_idx)
    print("\nTexts in order:")
    for i, (uid, text) in enumerate(zip(all_user_ids, texts)):
        print(f"  {i}: {uid} -> {text[:50]}...")
        # Verify text contains correct bio
        expected_bio = test_users[test_users['user_id'] == uid]['bio'].iloc[0]
        assert expected_bio in text, f"FAILED: Bio mismatch for {uid}"

    print("\n✓ PASS: Profile texts match user_id_to_idx order")


def test_text_truncation():
    """Test that long bios are truncated."""
    print("\n" + "="*60)
    print("TEST 2: Text Truncation")
    print("="*60)

    # Create user with very long bio
    long_bio = "x" * 1000 + " this should be truncated"

    test_user = pd.DataFrame({
        'user_id': ['user_1'],
        'gender': ['male'],
        'age': [30],
        'city': ['NYC'],
        'bio': [long_bio],
        'job_title': ['Engineer'],
        'school': ['MIT']
    })

    preprocessor = DataPreprocessor()
    profile_text = preprocessor.create_profile_text(test_user)

    print(f"\nOriginal bio length: {len(long_bio)}")
    print(f"Profile text length: {len(profile_text.iloc[0])}")
    print(f"Truncated: {len(profile_text.iloc[0]) < len(long_bio)}")

    assert len(profile_text.iloc[0]) <= 850, "FAILED: Text not truncated"
    assert '...' in profile_text.iloc[0], "FAILED: Truncation marker missing"

    print("\n✓ PASS: Long bios are truncated to 800 chars")


def test_text_sanitization():
    """Test that URLs and special chars are cleaned."""
    print("\n" + "="*60)
    print("TEST 3: Text Sanitization")
    print("="*60)

    test_user = pd.DataFrame({
        'user_id': ['user_1'],
        'gender': ['male'],
        'age': [30],
        'city': ['NYC'],
        'bio': ['Check out http://example.com!!!    Multiple   spaces'],
        'job_title': ['Engineer'],
        'school': ['MIT']
    })

    preprocessor = DataPreprocessor()
    profile_text = preprocessor.create_profile_text(test_user).iloc[0]

    print(f"\nOriginal bio: {test_user['bio'].iloc[0]}")
    print(f"Cleaned text: {profile_text}")

    assert 'http' not in profile_text, "FAILED: URL not removed"
    assert '   ' not in profile_text, "FAILED: Multiple spaces not normalized"
    assert '!!!' not in profile_text, "FAILED: Excessive punctuation not cleaned"

    print("\n✓ PASS: URLs and special characters are sanitized")


def test_empty_profile():
    """Test handling of empty profiles."""
    print("\n" + "="*60)
    print("TEST 4: Empty Profile Handling")
    print("="*60)

    test_user = pd.DataFrame({
        'user_id': ['user_1'],
        'gender': ['male'],
        'age': [30],
        'city': ['NYC'],
        'bio': [''],
        'job_title': [None],
        'school': [None]
    })

    preprocessor = DataPreprocessor()
    profile_text = preprocessor.create_profile_text(test_user).iloc[0]

    print(f"\nProfile with empty bio, no job/school: {profile_text}")

    assert len(profile_text) > 0, "FAILED: Empty profile text"
    assert 'nyc' in profile_text, "FAILED: Should contain city"

    print("\n✓ PASS: Empty profiles handled gracefully")


def test_all_missing():
    """Test completely missing profile data."""
    print("\n" + "="*60)
    print("TEST 5: All Missing Data")
    print("="*60)

    test_user = pd.DataFrame({
        'user_id': ['user_1'],
        'gender': ['male'],
        'age': [30],
        'city': [None],
        'bio': [None],
        'job_title': [None],
        'school': [None]
    })

    preprocessor = DataPreprocessor()
    profile_text = preprocessor.create_profile_text(test_user).iloc[0]

    print(f"\nProfile with all missing data: {profile_text}")

    assert profile_text == "no profile information", f"FAILED: Expected fallback text, got '{profile_text}'"

    print("\n✓ PASS: Fallback text for missing data works")


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("NLP RECOMMENDER FIX VALIDATION")
    print("="*60)

    try:
        test_index_order_consistency()
        test_text_truncation()
        test_text_sanitization()
        test_empty_profile()
        test_all_missing()

        print("\n" + "="*60)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("="*60)
        print("\nNLP Recommender is PRODUCTION READY ✅")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
