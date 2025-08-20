import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ai_model():
    print("🧪 Testing AI Model Manager...")
    print("-" * 50)
    
    try:
        # Test 1: Import the model manager
        print("📦 Testing import...")
        from app.models.ai_models import ai_model_manager
        print("✅ Import successful")
        
        # Test 2: Load the model
        print("\n🤖 Testing model loading...")
        model = ai_model_manager.get_model()
        print(f"✅ Model loaded successfully: {type(model)}")
        print(f"   Model name: {model.get_sentence_embedding_dimension()} dimensions")
        
        # Test 3: Simple similarity test
        print("\n📊 Testing similarity calculation...")
        cv_text = "I am a Python developer with 3 years of experience in Django and FastAPI"
        job_text = "We are looking for a Python developer with Django experience"
        
        similarity = ai_model_manager.calculate_similarity(cv_text, job_text)
        print(f"✅ Similarity calculated: {similarity:.4f} ({similarity*100:.2f}%)")
        
        # Test 4: Multiple similarity tests
        print("\n🔍 Testing multiple scenarios...")
        
        test_cases = [
            {
                "cv": "Java developer with Spring Boot experience",
                "job": "Looking for Java Spring developer",
                "expected": "High similarity"
            },
            {
                "cv": "Marketing specialist with social media expertise",
                "job": "Software engineer position requiring Python",
                "expected": "Low similarity"
            },
            {
                "cv": "Data scientist with machine learning and Python skills",
                "job": "Data scientist role focusing on ML and AI",
                "expected": "High similarity"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            similarity = ai_model_manager.calculate_similarity(test["cv"], test["job"])
            print(f"   Test {i}: {similarity:.4f} ({similarity*100:.2f}%) - {test['expected']}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ai_model()
    if success:
        print("\n✨ Phase 2 completed successfully! Ready for Phase 3.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")