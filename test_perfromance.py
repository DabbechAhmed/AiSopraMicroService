import requests
import time
import asyncio
import aiohttp
import concurrent.futures
from typing import List

class PerformanceOptimizer:
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1/similarity"
        self.sync_api_url = f"{base_url}/api/v1/similarity/sync"
    
    def test_sync_requests(self, num_requests=10):
        """Test synchronous requests performance"""
        print(f"\n🔄 Testing {num_requests} synchronous requests...")
        
        data = {"cv_text": "Python developer", "job_text": "Python position"}
        times = []
        
        # Warm up
        requests.post(self.api_url, json=data)
        
        for i in range(num_requests):
            start = time.time()
            response = requests.post(self.api_url, json=data)
            duration = (time.time() - start) * 1000
            times.append(duration)
            
            if response.status_code == 200:
                result = response.json()
                processing_time = result.get("processing_time_ms", 0)
                print(f"  Request {i+1}: {duration:.1f}ms (processing: {processing_time:.1f}ms)")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"  📊 Average: {avg_time:.1f}ms, Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")
        return avg_time
    
    async def test_async_requests(self, num_requests=10):
        """Test asynchronous requests performance"""
        print(f"\n⚡ Testing {num_requests} asynchronous requests...")
        
        data = {"cv_text": "Python developer", "job_text": "Python position"}
        
        async def make_request(session, request_id):
            start = time.time()
            async with session.post(self.api_url, json=data) as response:
                result = await response.json()
                duration = (time.time() - start) * 1000
                processing_time = result.get("processing_time_ms", 0)
                print(f"  Request {request_id}: {duration:.1f}ms (processing: {processing_time:.1f}ms)")
                return duration
        
        async with aiohttp.ClientSession() as session:
            # Warm up
            await make_request(session, 0)
            
            # Concurrent requests
            tasks = [make_request(session, i+1) for i in range(num_requests)]
            times = await asyncio.gather(*tasks)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"  📊 Average: {avg_time:.1f}ms, Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")
        return avg_time
    
    def test_threading(self, num_requests=10):
        """Test with threading"""
        print(f"\n🧵 Testing {num_requests} threaded requests...")
        
        data = {"cv_text": "Python developer", "job_text": "Python position"}
        
        def make_request():
            start = time.time()
            response = requests.post(self.api_url, json=data)
            duration = (time.time() - start) * 1000
            return duration
        
        # Warm up
        make_request()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            times = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        avg_time = sum(times) / len(times)
        print(f"  📊 Average: {avg_time:.1f}ms")
        return avg_time
    
    async def run_all_tests(self):
        """Run comprehensive performance tests"""
        print("🧪 COMPREHENSIVE PERFORMANCE TEST")
        print("=" * 50)
        
        # Test 1: Sync requests
        sync_time = self.test_sync_requests(5)
        
        # Test 2: Async requests
        async_time = await self.test_async_requests(5)
        
        # Test 3: Threaded requests
        thread_time = self.test_threading(5)
        
        print("\n📊 SUMMARY")
        print("-" * 30)
        print(f"Synchronous:  {sync_time:.1f}ms")
        print(f"Asynchronous: {async_time:.1f}ms")
        print(f"Threaded:     {thread_time:.1f}ms")
        
        best_method = min([
            ("Synchronous", sync_time),
            ("Asynchronous", async_time), 
            ("Threaded", thread_time)
        ], key=lambda x: x[1])
        
        print(f"\n🏆 Best Method: {best_method[0]} ({best_method[1]:.1f}ms)")
        
        if best_method[1] < 100:
            print("🚀 EXCELLENT Performance!")
        elif best_method[1] < 500:
            print("✅ GOOD Performance!")
        else:
            print("⚠️ Performance needs improvement")

async def main():
    # Install aiohttp if not installed
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp...")
        import subprocess
        subprocess.check_call(["pip", "install", "aiohttp"])
        import aiohttp
    
    optimizer = PerformanceOptimizer()
    await optimizer.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())