import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import FracturePredictor

class TestFracturePredictor(unittest.TestCase):
    
    @patch('predictor.YOLO')
    @patch('os.path.exists')
    def test_initialization(self, mock_exists, mock_yolo):
        mock_exists.return_value = True
        predictor = FracturePredictor("dummy_path.pt")
        self.assertIsNotNone(predictor.model)
        mock_yolo.assert_called_with("dummy_path.pt")

    @patch('predictor.YOLO')
    @patch('os.path.exists')
    def test_predict_calls_model(self, mock_exists, mock_yolo):
        mock_exists.return_value = True
        
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance
        
        predictor = FracturePredictor("dummy_path.pt")
        dummy_image = MagicMock()
        
        predictor.predict(dummy_image, conf_threshold=0.5)
        
        mock_model_instance.assert_called_with(dummy_image, conf=0.5, iou=0.45)

if __name__ == '__main__':
    unittest.main()
