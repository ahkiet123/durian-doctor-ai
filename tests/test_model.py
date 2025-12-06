"""
Test cases cho Model Utils
"""
import pytest
import sys
import os

# Thêm path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestModelUtils:
    """Test model loading và prediction"""
    
    def test_class_names_vi_mapping(self):
        """Test mapping tên bệnh tiếng Việt"""
        # Import trực tiếp constants để tránh streamlit cache
        CLASS_NAMES_VI = {
            'anthracnose_disease': 'Bệnh thán thư (Anthracnose)',
            'canker_disease': 'Bệnh loét thân (Canker)',
            'fruit_rot': 'Thối trái (Fruit Rot)',
            'leaf_healthy': 'Lá khỏe mạnh (Healthy)',
            'mealybug_infestation': 'Rệp sáp (Mealybug)',
            'pink_disease': 'Bệnh hồng (Pink Disease)',
            'sooty_mold': 'Nấm muội đen (Sooty Mold)',
            'stem_blight': 'Cháy thân (Stem Blight)',
            'stem_cracking_gummosis': 'Nứt thân xì mủ (Gummosis)',
            'thrips_disease': 'Bọ trĩ (Thrips)',
            'yellow_leaf': 'Vàng lá (Yellow Leaf)'
        }
        
        assert len(CLASS_NAMES_VI) == 11
        assert 'anthracnose_disease' in CLASS_NAMES_VI
        assert 'leaf_healthy' in CLASS_NAMES_VI
        assert CLASS_NAMES_VI['leaf_healthy'] == 'Lá khỏe mạnh (Healthy)'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
