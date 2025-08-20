import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.ai_models import ai_model_manager

def test_ai_model():
    try:
        # Test model loading
        model = ai_model_manager.get_model()
        print(f"✅ Model loaded successfully: {type(model)}")
        
        # Test similarity calculation
        cv_text = "I am a Python developer with 3 years of experience in Django and FastAPI"
        job_text = "We are looking for a Python developer with Django experience"
        
        similarity = ai_model_manager.calculate_similarity(cv_text, job_text)
        print(f"✅ Similarity calculated: {similarity:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_ai_model()