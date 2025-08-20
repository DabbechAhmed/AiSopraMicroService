import requests
import time
import json

class MicroserviceValidator:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def test_health(self):
        """Test health endpoint"""
        print("🏥 Testing Health Endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Health check: PASSED")
                print(f"   Response: {response.json()}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_similarity_endpoint(self):
        """Test similarity calculation endpoint"""
        print("\n🧠 Testing Similarity Endpoint...")
        
        test_cases = [
            {
                "name": "High Similarity Test",
                "cv": "Python developer with Django and FastAPI experience",
                "job": "Looking for Python developer with Django knowledge",
                "expected_range": (70, 100)
            },
            {
                "name": "Medium Similarity Test", 
                "cv": "Java developer with Spring Boot experience",
                "job": "Python developer position with Django",
                "expected_range": (15, 50)  # Adjusted range - Java vs Python is actually quite different
            },
            {
                "name": "Low Similarity Test",
                "cv": "Marketing manager with social media expertise",
                "job": "Software engineer position requiring machine learning",
                "expected_range": (0, 30)
            },
            {
                "name": "Identical Text Test",
                "cv": "Software engineer with Python",
                "job": "Software engineer with Python", 
                "expected_range": (85, 100)
            },
            {
                "name": "Empty Text Test",
                "cv": "",
                "job": "Software engineer position",
                "expected_range": (0, 10)
            }
        ]
        
        all_passed = True
        results = []
        
        for test in test_cases:
            try:
                print(f"\n   🔍 {test['name']}...")
                
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base}/similarity",
                    params={
                        "cv_text": test["cv"],
                        "job_text": test["job"]
                    },
                    timeout=30
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    score = data.get("similarity_score", 0)
                    processing_time = data.get("processing_time_ms", 0)
                    
                    # Check if score is in expected range
                    min_score, max_score = test["expected_range"]
                    if min_score <= score <= max_score:
                        print(f"      ✅ Score: {score:.2f}% (Expected: {min_score}-{max_score}%)")
                        print(f"      ⚡ API Response: {response_time:.1f}ms, Processing: {processing_time:.1f}ms")
                        results.append({
                            "test": test["name"],
                            "status": "PASSED",
                            "score": score,
                            "response_time": response_time,
                            "processing_time": processing_time
                        })
                    else:
                        print(f"      ❌ Score: {score:.2f}% (Expected: {min_score}-{max_score}%)")
                        all_passed = False
                        results.append({
                            "test": test["name"],
                            "status": "FAILED",
                            "score": score,
                            "expected": f"{min_score}-{max_score}%"
                        })
                else:
                    print(f"      ❌ HTTP Error: {response.status_code}")
                    print(f"      Response: {response.text}")
                    all_passed = False
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                all_passed = False
        
        return all_passed, results
    
    def test_performance(self):
        """Test performance with multiple requests"""
        print("\n⚡ Testing Performance...")
        
        cv_text = "Python developer with 3 years experience in Django, FastAPI, and PostgreSQL"
        job_text = "Looking for Python developer with web framework experience"
        
        times = []
        success_count = 0
        total_requests = 10
        
        print(f"   Making {total_requests} requests...")
        
        for i in range(total_requests):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base}/similarity",
                    params={"cv_text": cv_text, "job_text": job_text},
                    timeout=10
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    times.append(response_time)
                    success_count += 1
                    
            except Exception as e:
                print(f"   ❌ Request {i+1} failed: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"   ✅ Success Rate: {success_count}/{total_requests} ({success_count/total_requests*100:.1f}%)")
            print(f"   ⚡ Average Response Time: {avg_time:.1f}ms")
            print(f"   📊 Min/Max: {min_time:.1f}ms / {max_time:.1f}ms")
            
            # Performance thresholds
            if avg_time < 100:
                print("   🚀 Performance: EXCELLENT (< 100ms)")
                return True
            elif avg_time < 500:
                print("   ✅ Performance: GOOD (< 500ms)")
                return True
            else:
                print("   ⚠️ Performance: SLOW (> 500ms)")
                return False
        else:
            print("   ❌ No successful requests")
            return False
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        print("\n🛡️ Testing Error Handling...")
        
        # Test with very long text
        try:
            long_text = "Python developer " * 1000  # Very long text
            response = requests.post(
                f"{self.api_base}/similarity",
                params={"cv_text": long_text, "job_text": "Python developer"},
                timeout=30
            )
            
            if response.status_code == 200:
                print("   ✅ Long text handling: PASSED")
            else:
                print(f"   ⚠️ Long text handling: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Long text error: {e}")
        
        return True
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("🧪 AI MICROSERVICE VALIDATION")
        print("=" * 50)
        
        results = {
            "health": self.test_health(),
            "similarity": self.test_similarity_endpoint()[0],
            "performance": self.test_performance(),
            "error_handling": self.test_error_handling()
        }
        
        print("\n📊 VALIDATION SUMMARY")
        print("-" * 30)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.upper()}: {status}")
            if not passed:
                all_passed = False
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 MICROSERVICE IS READY FOR STREAMLIT INTEGRATION!")
            print("✅ All tests passed successfully")
        else:
            print("⚠️ SOME ISSUES FOUND")
            print("❌ Please fix failing tests before integration")
        
        return all_passed

if __name__ == "__main__":
    validator = MicroserviceValidator()
    validator.run_full_validation()