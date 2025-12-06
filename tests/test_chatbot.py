"""
Test cases cho chatbot keywords matching
"""
import pytest


# Danh sách từ khóa liên quan đến sầu riêng
DURIAN_KEYWORDS = [
    'sầu riêng', 'durian', 'bệnh', 'lá', 'trái', 'thân', 'rễ', 'thuốc', 'phân', 'bón', 
    'phun', 'trị', 'chữa', 'triệu chứng', 'vàng', 'thối', 'nấm', 'sâu', 
    'côn trùng', 'rệp', 'nhện', 'xì mủ', 'nứt', 'cháy', 'héo', 'chăm sóc',
    'tưới', 'cắt tỉa', 'ra hoa', 'đậu trái', 'thu hoạch', 'giống', 'ri6', 'monthong', 'thái',
    'cây', 'vườn', 'nhà vườn', 'nông dân', 'thương lái', 'musang', 'dona'
]


def is_durian_related(query: str) -> bool:
    """Kiểm tra query có liên quan đến sầu riêng không"""
    query_lower = query.lower()
    return any(kw in query_lower for kw in DURIAN_KEYWORDS)


class TestKeywordMatching:
    """Test keyword matching cho RAG trigger"""
    
    def test_durian_questions_should_trigger_rag(self):
        """Các câu hỏi về sầu riêng phải trigger RAG"""
        queries = [
            "Bệnh thán thư trên sầu riêng",
            "Cách bón phân cho cây",
            "Lá bị vàng là do gì?",
            "Thuốc trị nấm cho sầu riêng",
            "Giống Ri6 có tốt không?",
            "Sầu riêng Thái bao nhiêu ngày thu hoạch?",
        ]
        for q in queries:
            assert is_durian_related(q), f"Expected TRUE for: {q}"
    
    def test_unrelated_questions_should_not_trigger_rag(self):
        """Các câu hỏi không liên quan phải KHÔNG trigger RAG"""
        queries = [
            "hello",
            "xin chào",
            "thuật toán BFS là gì?",
            "code Python như thế nào?",
            "thời tiết hôm nay",
            "giá bitcoin bao nhiêu?",
            "tổng thống Mỹ là ai?",
        ]
        for q in queries:
            assert not is_durian_related(q), f"Expected FALSE for: {q}"
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Câu dài nhưng không liên quan
        assert not is_durian_related("Tôi muốn hỏi về cách làm bánh mì ngon nhất thế giới")
        
        # Câu ngắn nhưng liên quan
        assert is_durian_related("lá vàng")
        assert is_durian_related("bệnh nấm")
        
        # Chữ hoa/thường
        assert is_durian_related("SẦU RIÊNG")
        assert is_durian_related("BỆNH")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
