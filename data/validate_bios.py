"""
Bio Quality Validation Script

Validates the quality of generated user bios with comprehensive checks:
- Duplicate detection
- Length distribution
- Vocabulary diversity
- Grammatical issues
- Content quality
- Semantic diversity

Outputs:
- Console report with quality score
- JSON report with detailed findings
- Sample bios for manual inspection
- Actionable recommendations
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import warnings

# NLP libraries (optional for semantic analysis)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"⚠️  Semantic diversity check disabled (missing dependencies: {e})")
    SEMANTIC_AVAILABLE = False
    SentenceTransformer = None
    cosine_similarity = None

warnings.filterwarnings('ignore')


class BioValidator:
    """Validates quality of generated user bios."""

    # Quality thresholds
    DUPLICATE_THRESHOLD = 20  # Max users per bio before flagging
    MIN_LENGTH = 20  # Minimum characters
    MAX_LENGTH = 500  # Maximum characters
    SIMILARITY_CLUSTER_THRESHOLD = 0.95  # Very similar bios
    TARGET_UNIQUE_RATIO = 0.3  # Vocabulary: unique/total words

    # Stop words (common words to exclude from analysis)
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its',
        'our', 'their', 'this', 'that', 'these', 'those', 'am'
    }

    def __init__(self, users_csv_path: str = 'data/raw/users.csv'):
        """
        Initialize bio validator.

        Args:
            users_csv_path: Path to users.csv file
        """
        self.users_csv_path = users_csv_path
        self.users_df = None
        self.bios = []
        self.validation_results = {}
        self.quality_score = 0

    def load_data(self) -> None:
        """Load users data and extract bios."""
        print(f"Loading data from {self.users_csv_path}...")

        self.users_df = pd.read_csv(self.users_csv_path)

        if 'bio' not in self.users_df.columns:
            raise ValueError("No 'bio' column found in users.csv")

        # Extract non-null bios
        self.bios = self.users_df['bio'].dropna().tolist()

        print(f"✓ Loaded {len(self.users_df)} users")
        print(f"✓ Found {len(self.bios)} non-null bios")

    def check_duplicates(self) -> Dict:
        """
        Check for duplicate bios.

        Returns:
            Dictionary with duplicate statistics
        """
        print("\n[1/6] Checking for duplicates...")

        bio_counts = Counter(self.bios)

        # Find duplicates
        duplicates = {bio: count for bio, count in bio_counts.items() if count > 1}
        exact_duplicates = len(duplicates)

        # Find high-frequency bios (>20 users)
        high_freq_bios = {bio: count for bio, count in duplicates.items()
                          if count > self.DUPLICATE_THRESHOLD}

        # Most common bios
        most_common = bio_counts.most_common(10)

        results = {
            'total_unique_bios': len(bio_counts),
            'total_bios': len(self.bios),
            'unique_ratio': len(bio_counts) / len(self.bios),
            'exact_duplicates': exact_duplicates,
            'high_frequency_count': len(high_freq_bios),
            'high_frequency_bios': [
                {'bio': bio[:100], 'count': count}
                for bio, count in sorted(high_freq_bios.items(), key=lambda x: x[1], reverse=True)[:5]
            ],
            'most_common': [
                {'bio': bio[:100], 'count': count}
                for bio, count in most_common
            ]
        }

        print(f"  Total unique bios: {results['total_unique_bios']}")
        print(f"  Unique ratio: {results['unique_ratio']:.2%}")
        print(f"  Bios shared by >1 user: {exact_duplicates}")
        print(f"  Bios shared by >{self.DUPLICATE_THRESHOLD} users: {len(high_freq_bios)}")

        if high_freq_bios:
            print(f"  ⚠️  WARNING: {len(high_freq_bios)} bios used by >{self.DUPLICATE_THRESHOLD} users")

        return results

    def check_length_distribution(self) -> Dict:
        """
        Analyze bio length distribution.

        Returns:
            Dictionary with length statistics
        """
        print("\n[2/6] Analyzing length distribution...")

        lengths = [len(bio) for bio in self.bios]

        # Find problematic lengths
        too_short = [bio for bio in self.bios if len(bio) < self.MIN_LENGTH]
        too_long = [bio for bio in self.bios if len(bio) > self.MAX_LENGTH]

        # Word counts
        word_counts = [len(bio.split()) for bio in self.bios]

        results = {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'too_short_count': len(too_short),
            'too_long_count': len(too_long),
            'too_short_examples': too_short[:3],
            'too_long_examples': [bio[:200] + '...' for bio in too_long[:3]],
            'mean_words': np.mean(word_counts),
            'median_words': np.median(word_counts)
        }

        print(f"  Character length: min={results['min_length']}, max={results['max_length']}, "
              f"mean={results['mean_length']:.1f}, median={results['median_length']:.0f}")
        print(f"  Word count: mean={results['mean_words']:.1f}, median={results['median_words']:.0f}")

        if too_short:
            print(f"  ⚠️  WARNING: {len(too_short)} bios shorter than {self.MIN_LENGTH} chars")

        if too_long:
            print(f"  ⚠️  WARNING: {len(too_long)} bios longer than {self.MAX_LENGTH} chars")

        return results

    def check_vocabulary_diversity(self) -> Dict:
        """
        Analyze vocabulary diversity.

        Returns:
            Dictionary with vocabulary statistics
        """
        print("\n[3/6] Analyzing vocabulary diversity...")

        # Extract all words
        all_words = []
        for bio in self.bios:
            words = re.findall(r'\b[a-zA-Z]+\b', bio.lower())
            all_words.extend(words)

        # Count words
        word_counts = Counter(all_words)
        total_words = len(all_words)
        unique_words = len(word_counts)

        # Exclude stop words for content analysis
        content_words = [w for w in all_words if w not in self.STOP_WORDS]
        content_word_counts = Counter(content_words)

        # Hapax legomena (words appearing only once)
        hapax = [word for word, count in word_counts.items() if count == 1]

        # Most common content words
        most_common_content = content_word_counts.most_common(20)

        results = {
            'total_words': total_words,
            'unique_words': unique_words,
            'unique_ratio': unique_words / total_words if total_words > 0 else 0,
            'hapax_count': len(hapax),
            'hapax_ratio': len(hapax) / unique_words if unique_words > 0 else 0,
            'most_common_words': [
                {'word': word, 'count': count, 'frequency': count / total_words}
                for word, count in most_common_content
            ]
        }

        print(f"  Total words: {total_words:,}")
        print(f"  Unique words: {unique_words:,}")
        print(f"  Unique ratio: {results['unique_ratio']:.2%} "
              f"(target: >{self.TARGET_UNIQUE_RATIO:.0%})")
        print(f"  Hapax legomena: {len(hapax):,} ({results['hapax_ratio']:.2%})")

        if results['unique_ratio'] < self.TARGET_UNIQUE_RATIO:
            print(f"  ⚠️  WARNING: Vocabulary diversity below target "
                  f"({results['unique_ratio']:.2%} < {self.TARGET_UNIQUE_RATIO:.0%})")

        return results

    def check_grammatical_issues(self) -> Dict:
        """
        Check for grammatical and formatting issues.

        Returns:
            Dictionary with issue counts and examples
        """
        print("\n[4/6] Checking for grammatical issues...")

        issues = {
            'no_punctuation': [],
            'double_spaces': [],
            'double_punctuation': [],
            'weird_capitalization': [],
            'starts_lowercase': [],
            'missing_periods': []
        }

        for bio in self.bios:
            # No punctuation at all
            if not re.search(r'[.!?,;:]', bio):
                issues['no_punctuation'].append(bio)

            # Double spaces
            if '  ' in bio:
                issues['double_spaces'].append(bio)

            # Double punctuation (except ellipsis)
            if re.search(r'[.!?,;:]{2,}', bio.replace('...', '')):
                issues['double_punctuation'].append(bio)

            # Starts with lowercase (except special cases like 'iPhone')
            if bio and bio[0].islower():
                issues['starts_lowercase'].append(bio)

            # Weird capitalization (multiple uppercase in middle of word)
            if re.search(r'\b[a-z]+[A-Z]+[a-z]+', bio):
                issues['weird_capitalization'].append(bio)

            # Missing period at end of sentence
            if bio and not bio.rstrip().endswith(('.', '!', '?')):
                issues['missing_periods'].append(bio)

        results = {
            issue_type: {
                'count': len(examples),
                'percentage': len(examples) / len(self.bios) * 100,
                'examples': examples[:3]
            }
            for issue_type, examples in issues.items()
        }

        # Summary
        total_issues = sum(len(examples) for examples in issues.values())

        print(f"  Total grammatical issues: {total_issues}")

        for issue_type, data in results.items():
            if data['count'] > 0:
                print(f"    {issue_type}: {data['count']} ({data['percentage']:.1f}%)")

        return results

    def check_content_quality(self) -> Dict:
        """
        Check for content quality issues.

        Returns:
            Dictionary with quality issues
        """
        print("\n[5/6] Checking content quality...")

        issues = {
            'null_or_empty': [],
            'placeholder_text': [],
            'numeric_only': [],
            'special_chars_only': [],
            'too_repetitive': []
        }

        # Placeholder patterns
        placeholder_patterns = [
            r'\b(test|example|user|placeholder|lorem|ipsum|dummy)\b'
        ]

        for bio in self.bios:
            # Check for placeholders
            for pattern in placeholder_patterns:
                if re.search(pattern, bio, re.IGNORECASE):
                    issues['placeholder_text'].append(bio)
                    break

            # Numeric only
            if re.match(r'^[\d\s]+$', bio):
                issues['numeric_only'].append(bio)

            # Special characters only
            if re.match(r'^[^a-zA-Z0-9\s]+$', bio):
                issues['special_chars_only'].append(bio)

            # Too repetitive (same word 5+ times)
            words = bio.lower().split()
            word_counts = Counter(words)
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count >= 5:
                issues['too_repetitive'].append(bio)

        # Check for null/empty in original dataframe
        null_count = self.users_df['bio'].isna().sum()
        empty_count = (self.users_df['bio'] == '').sum()

        results = {
            'null_count': int(null_count),
            'empty_count': int(empty_count),
            'issues': {
                issue_type: {
                    'count': len(examples),
                    'percentage': len(examples) / len(self.bios) * 100,
                    'examples': examples[:3]
                }
                for issue_type, examples in issues.items()
            }
        }

        print(f"  Null bios: {null_count}")
        print(f"  Empty bios: {empty_count}")

        for issue_type, data in results['issues'].items():
            if data['count'] > 0:
                print(f"  {issue_type}: {data['count']} ({data['percentage']:.1f}%)")
                if data['count'] > 0:
                    print(f"    ⚠️  WARNING: Found {issue_type} issues")

        return results

    def check_semantic_diversity(self) -> Dict:
        """
        Analyze semantic diversity using sentence embeddings.

        Returns:
            Dictionary with semantic similarity statistics
        """
        print("\n[6/6] Analyzing semantic diversity...")

        # Skip if dependencies not available
        if not SEMANTIC_AVAILABLE:
            print("  ⚠️  Skipping semantic diversity check (dependencies not installed)")
            return {
                'sample_size': 0,
                'mean_similarity': 0.0,
                'median_similarity': 0.0,
                'std_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'very_similar_pairs': 0,
                'very_similar_examples': [],
                'skipped': True
            }

        print("  Loading sentence transformer model...")

        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Sample bios for performance (full analysis on 20K bios takes ~5 min)
        sample_size = min(1000, len(self.bios))
        sample_bios = np.random.choice(self.bios, size=sample_size, replace=False)

        print(f"  Generating embeddings for {sample_size} bios...")
        embeddings = model.encode(sample_bios, batch_size=64, show_progress_bar=True)

        print("  Computing pairwise similarities...")
        similarity_matrix = cosine_similarity(embeddings)

        # Get upper triangle (avoid duplicates and self-similarity)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        # Find very similar bio pairs
        very_similar_pairs = []
        for i in range(len(sample_bios)):
            for j in range(i + 1, len(sample_bios)):
                sim = similarity_matrix[i, j]
                if sim > self.SIMILARITY_CLUSTER_THRESHOLD:
                    very_similar_pairs.append({
                        'bio1': sample_bios[i][:100],
                        'bio2': sample_bios[j][:100],
                        'similarity': float(sim)
                    })

        results = {
            'sample_size': sample_size,
            'mean_similarity': float(np.mean(upper_triangle)),
            'median_similarity': float(np.median(upper_triangle)),
            'std_similarity': float(np.std(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle)),
            'very_similar_pairs': len(very_similar_pairs),
            'very_similar_examples': very_similar_pairs[:5]
        }

        print(f"  Mean pairwise similarity: {results['mean_similarity']:.3f}")
        print(f"  Median pairwise similarity: {results['median_similarity']:.3f}")
        print(f"  Very similar pairs (>{self.SIMILARITY_CLUSTER_THRESHOLD}): "
              f"{results['very_similar_pairs']}")

        if results['mean_similarity'] > 0.8:
            print(f"  ⚠️  WARNING: High average similarity ({results['mean_similarity']:.3f}) "
                  f"indicates low semantic diversity")

        return results

    def calculate_quality_score(self) -> int:
        """
        Calculate overall quality score (0-100).

        Returns:
            Quality score
        """
        score = 100
        penalties = []

        # Duplicate penalty (max -30 points)
        unique_ratio = self.validation_results['duplicates']['unique_ratio']
        if unique_ratio < 0.5:
            penalty = (0.5 - unique_ratio) * 60  # -30 points at 0% unique
            score -= penalty
            penalties.append(f"Low uniqueness: -{penalty:.1f}")

        # Length issues penalty (max -10 points)
        length_stats = self.validation_results['length']
        issue_ratio = (length_stats['too_short_count'] + length_stats['too_long_count']) / len(self.bios)
        if issue_ratio > 0.01:
            penalty = min(issue_ratio * 100, 10)
            score -= penalty
            penalties.append(f"Length issues: -{penalty:.1f}")

        # Vocabulary diversity penalty (max -20 points)
        vocab_ratio = self.validation_results['vocabulary']['unique_ratio']
        if vocab_ratio < self.TARGET_UNIQUE_RATIO:
            penalty = (self.TARGET_UNIQUE_RATIO - vocab_ratio) * 66  # -20 points at 0%
            score -= penalty
            penalties.append(f"Low vocabulary diversity: -{penalty:.1f}")

        # Grammatical issues penalty (max -15 points)
        grammar = self.validation_results['grammar']
        total_grammar_issues = sum(data['count'] for data in grammar.values())
        grammar_issue_ratio = total_grammar_issues / len(self.bios)
        if grammar_issue_ratio > 0.05:
            penalty = min(grammar_issue_ratio * 50, 15)
            score -= penalty
            penalties.append(f"Grammar issues: -{penalty:.1f}")

        # Content quality penalty (max -15 points)
        content = self.validation_results['content']
        total_content_issues = sum(data['count'] for data in content['issues'].values())
        content_issue_ratio = total_content_issues / len(self.bios)
        if content_issue_ratio > 0.01:
            penalty = min(content_issue_ratio * 100, 15)
            score -= penalty
            penalties.append(f"Content quality issues: -{penalty:.1f}")

        # Semantic diversity penalty (max -10 points)
        semantic = self.validation_results['semantic']
        if not semantic.get('skipped', False) and semantic['mean_similarity'] > 0.8:
            penalty = (semantic['mean_similarity'] - 0.8) * 50  # -10 points at 1.0
            score -= penalty
            penalties.append(f"Low semantic diversity: -{penalty:.1f}")

        score = max(0, int(score))

        return score, penalties

    def generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations for improvement.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Duplicates
        unique_ratio = self.validation_results['duplicates']['unique_ratio']
        if unique_ratio < 0.7:
            needed = int((0.7 - unique_ratio) * len(self.bios))
            recommendations.append(
                f"⚠️  CRITICAL: Add {needed:,} more unique bio variations to reach 70% uniqueness. "
                f"Suggestion: Create 3-5 new template categories with 50+ variations each."
            )
        elif unique_ratio < 0.85:
            recommendations.append(
                f"ℹ️  Improve uniqueness from {unique_ratio:.1%} to 85%+. "
                f"Suggestion: Add 2-3 more template variations or increase randomization."
            )

        # Length
        length_stats = self.validation_results['length']
        if length_stats['too_short_count'] > 0:
            recommendations.append(
                f"⚠️  Fix {length_stats['too_short_count']} bios shorter than {self.MIN_LENGTH} chars. "
                f"Suggestion: Ensure all templates generate at least 2-3 sentences."
            )

        # Vocabulary
        vocab_ratio = self.validation_results['vocabulary']['unique_ratio']
        if vocab_ratio < self.TARGET_UNIQUE_RATIO:
            improvement_needed = int((self.TARGET_UNIQUE_RATIO - vocab_ratio) / vocab_ratio * 100)
            recommendations.append(
                f"⚠️  Increase vocabulary diversity by {improvement_needed}%. "
                f"Suggestion: Expand word lists in templates by 30-50% with synonyms and variations."
            )

        # Grammar
        grammar = self.validation_results['grammar']
        for issue_type, data in grammar.items():
            if data['count'] > len(self.bios) * 0.05:  # More than 5% affected
                recommendations.append(
                    f"⚠️  Fix {issue_type}: {data['count']} bios ({data['percentage']:.1f}%). "
                    f"Suggestion: Review and fix template formatting."
                )

        # Content quality
        content = self.validation_results['content']
        for issue_type, data in content['issues'].items():
            if data['count'] > 0:
                recommendations.append(
                    f"⚠️  Remove {issue_type}: {data['count']} bios. "
                    f"Suggestion: Add validation checks in generation code."
                )

        # Semantic diversity
        semantic = self.validation_results['semantic']
        if not semantic.get('skipped', False) and semantic['mean_similarity'] > 0.8:
            recommendations.append(
                f"ℹ️  Increase semantic diversity (current similarity: {semantic['mean_similarity']:.3f}). "
                f"Suggestion: Create more template categories with distinct themes."
            )

        # High-frequency bios
        high_freq = self.validation_results['duplicates']['high_frequency_count']
        if high_freq > 0:
            recommendations.append(
                f"⚠️  CRITICAL: {high_freq} bios are used by >{self.DUPLICATE_THRESHOLD} users. "
                f"Suggestion: These specific bios need immediate attention - add variations."
            )

        # Overall
        if not recommendations:
            recommendations.append(
                "✅ EXCELLENT: Bio quality meets all standards! No improvements needed."
            )

        return recommendations

    def print_report(self) -> None:
        """Print comprehensive validation report to console."""
        print("\n" + "="*90)
        print("BIO QUALITY VALIDATION REPORT")
        print("="*90)

        # Quality score
        score, penalties = self.calculate_quality_score()

        print(f"\nOVERALL QUALITY SCORE: {score}/100")

        if score >= 90:
            grade = "A (Excellent)"
        elif score >= 80:
            grade = "B (Good)"
        elif score >= 70:
            grade = "C (Fair)"
        elif score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Failed)"

        print(f"GRADE: {grade}")

        if penalties:
            print(f"\nPenalties applied:")
            for penalty in penalties:
                print(f"  - {penalty}")

        # Summary statistics
        print(f"\n" + "-"*90)
        print("SUMMARY STATISTICS")
        print("-"*90)
        print(f"Total bios: {len(self.bios):,}")
        print(f"Unique bios: {self.validation_results['duplicates']['total_unique_bios']:,}")
        print(f"Uniqueness ratio: {self.validation_results['duplicates']['unique_ratio']:.2%}")
        print(f"Average length: {self.validation_results['length']['mean_length']:.1f} chars")
        print(f"Vocabulary size: {self.validation_results['vocabulary']['unique_words']:,} words")

        semantic = self.validation_results['semantic']
        if not semantic.get('skipped', False):
            print(f"Semantic diversity: {1 - semantic['mean_similarity']:.2%}")
        else:
            print(f"Semantic diversity: N/A (check skipped)")

        # Recommendations
        recommendations = self.generate_recommendations()

        print(f"\n" + "-"*90)
        print(f"RECOMMENDATIONS ({len(recommendations)})")
        print("-"*90)

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")

        print("\n" + "="*90)

    def show_random_samples(self, n: int = 20) -> None:
        """
        Show random sample bios for manual inspection.

        Args:
            n: Number of samples to show
        """
        print(f"\n" + "="*90)
        print(f"RANDOM SAMPLE BIOS (Manual Inspection)")
        print("="*90)

        sample_bios = np.random.choice(self.bios, size=min(n, len(self.bios)), replace=False)

        for i, bio in enumerate(sample_bios, 1):
            print(f"\n{i}. {bio}")

        print("\n" + "="*90)

    def save_report(self, output_path: str = 'data/bio_validation_report.json') -> None:
        """
        Save detailed validation results to JSON.

        Args:
            output_path: Path to save JSON report
        """
        report = {
            'summary': {
                'total_bios': len(self.bios),
                'quality_score': self.quality_score,
                'grade': self._get_grade(self.quality_score),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'validation_results': self.validation_results,
            'recommendations': self.generate_recommendations()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Detailed report saved to {output_path}")

    def _get_grade(self, score: int) -> str:
        """Get letter grade from score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def run_full_validation(self) -> None:
        """Run complete validation pipeline."""
        print("="*90)
        print("STARTING BIO QUALITY VALIDATION")
        print("="*90)

        # Load data
        self.load_data()

        # Run all checks
        self.validation_results['duplicates'] = self.check_duplicates()
        self.validation_results['length'] = self.check_length_distribution()
        self.validation_results['vocabulary'] = self.check_vocabulary_diversity()
        self.validation_results['grammar'] = self.check_grammatical_issues()
        self.validation_results['content'] = self.check_content_quality()
        self.validation_results['semantic'] = self.check_semantic_diversity()

        # Calculate quality score
        self.quality_score, _ = self.calculate_quality_score()

        # Print report
        self.print_report()

        # Show samples
        self.show_random_samples(n=20)

        # Save detailed report
        self.save_report()


def main():
    """Main execution."""
    validator = BioValidator()
    validator.run_full_validation()


if __name__ == '__main__':
    main()
