# app/mab_enhanced.py - ENHANCEMENT

class EnhancedContextExtractor:
    def extract(self, query: str, embedding: np.ndarray):
        """Rich context for better threshold selection"""
        return {
            'domain': self._detect_domain(query),  # existing
            'length_bin': self._length_bin(query),  # existing
            'complexity': self._query_complexity(query),  # NEW
            'entity_density': self._entity_count(query),  # NEW
            'uncertainty': self._embedding_uncertainty(embedding),  # NEW
            'temporal_relevance': self._temporal_score(query),  # NEW
        }
    
    def _query_complexity(self, q: str) -> str:
        # Simple vs compound vs multi-hop
        if 'and' in q.lower() and '?' in q:
            return 'compound'
        elif sum(c.isdigit() for c in q) > 5:
            return 'numerical'
        return 'simple'