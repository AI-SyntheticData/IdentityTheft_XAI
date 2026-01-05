"""
Test script for the Identity Theft Detection API
"""

import requests
import json


API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Model Info")
    print("="*60)

    response = requests.get(f"{API_BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_single_prediction_suspicious():
    """Test single prediction - suspicious case"""
    print("\n" + "="*60)
    print("TEST 3: Single Prediction (Suspicious)")
    print("="*60)

    data = {
        "age": 28,
        "doc_auth_score": 0.35,
        "face_match_score": 0.42,
        "liveness_score": 0.38,
        "ip_geo_match": 0,
        "vpn_or_tor": 1,
        "device_reuse_count": 7,
        "email_risk_score": 0.85,
        "phone_voip_flag": 1,
        "ssn_high_risk_flag": 1
    }

    response = requests.post(f"{API_BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Decision: {result['prediction']['decision']}")
        print(f"Risk Score: {result['prediction']['risk_score']:.4f}")
        print(f"Reasons: {result['prediction'].get('reasons', [])}")
    else:
        try:
            error_data = response.json()
            print(f"Error: {error_data}")
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
            print(f"Error: {response.status_code} - {response.text or 'No response content'}")

    return response.status_code == 200


def test_single_prediction_legitimate():
    """Test single prediction - legitimate case"""
    print("\n" + "="*60)
    print("TEST 4: Single Prediction (Legitimate)")
    print("="*60)

    data = {
        "age": 42,
        "doc_auth_score": 0.95,
        "face_match_score": 0.92,
        "liveness_score": 0.88,
        "ip_geo_match": 1,
        "vpn_or_tor": 0,
        "device_reuse_count": 1,
        "email_risk_score": 0.15,
        "phone_voip_flag": 0,
        "ssn_high_risk_flag": 0
    }

    response = requests.post(f"{API_BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Decision: {result['prediction']['decision']}")
        print(f"Risk Score: {result['prediction']['risk_score']:.4f}")
        print(f"Reasons: {result['prediction'].get('reasons', [])}")
    else:
        try:
            error_data = response.json()
            print(f"Error: {error_data}")
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
            print(f"Error: {response.status_code} - {response.text or 'No response content'}")

    return response.status_code == 200


def test_batch_prediction():
    """Test batch predictions"""
    print("\n" + "="*60)
    print("TEST 5: Batch Prediction")
    print("="*60)

    data = {
        "applications": [
            {
                "age": 28,
                "doc_auth_score": 0.35,
                "face_match_score": 0.42,
                "liveness_score": 0.38,
                "ip_geo_match": 0,
                "vpn_or_tor": 1,
                "device_reuse_count": 7,
                "email_risk_score": 0.85,
                "phone_voip_flag": 1,
                "ssn_high_risk_flag": 1
            },
            {
                "age": 42,
                "doc_auth_score": 0.95,
                "face_match_score": 0.92,
                "liveness_score": 0.88,
                "ip_geo_match": 1,
                "vpn_or_tor": 0,
                "device_reuse_count": 1,
                "email_risk_score": 0.15,
                "phone_voip_flag": 0,
                "ssn_high_risk_flag": 0
            },
            {
                "age": 35,
                "doc_auth_score": 0.65,
                "face_match_score": 0.70,
                "liveness_score": 0.68,
                "ip_geo_match": 1,
                "vpn_or_tor": 0,
                "device_reuse_count": 2,
                "email_risk_score": 0.45,
                "phone_voip_flag": 0,
                "ssn_high_risk_flag": 0
            }
        ],
        "explain": True
    }

    response = requests.post(f"{API_BASE_URL}/predict/batch", json=data)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Number of applications: {result['n_applications']}")
        print("\nDecisions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  App {i+1}: {pred['decision']} (Risk: {pred['risk_score']:.4f})")
    else:
        try:
            error_data = response.json()
            print(f"Error: {error_data}")
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
            print(f"Error: {response.status_code} - {response.text or 'No response content'}")

    return response.status_code == 200


def test_get_thresholds():
    """Test getting decision thresholds"""
    print("\n" + "="*60)
    print("TEST 6: Get Thresholds")
    print("="*60)

    response = requests.get(f"{API_BASE_URL}/thresholds")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_update_thresholds():
    """Test updating decision thresholds"""
    print("\n" + "="*60)
    print("TEST 7: Update Thresholds")
    print("="*60)

    data = {
        "threshold_low": 0.3,
        "threshold_high": 0.7
    }

    response = requests.post(f"{API_BASE_URL}/thresholds", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Restore original thresholds
    restore_data = {
        "threshold_low": 0.4,
        "threshold_high": 0.8
    }
    requests.post(f"{API_BASE_URL}/thresholds", json=restore_data)

    return response.status_code == 200


def test_missing_features():
    """Test error handling for missing features"""
    print("\n" + "="*60)
    print("TEST 8: Missing Features Error Handling")
    print("="*60)

    data = {
        "age": 35,
        "doc_auth_score": 0.75,
        # Missing other required features
    }

    response = requests.post(f"{API_BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Should return 400 Bad Request
    return response.status_code == 400


def main():
    """Run all API tests"""
    print("\n")
    print("*"*60)
    print("IDENTITY THEFT DETECTION API - TEST SUITE")
    print("*"*60)
    print("\nMake sure the API server is running:")
    print("  python src/api_server.py")
    print("\nPress Enter to continue...")
    input()

    tests = [
        ("Health Check", test_health),
        ("Model Info", test_info),
        ("Single Prediction (Suspicious)", test_single_prediction_suspicious),
        ("Single Prediction (Legitimate)", test_single_prediction_legitimate),
        ("Batch Prediction", test_batch_prediction),
        ("Get Thresholds", test_get_thresholds),
        ("Update Thresholds", test_update_thresholds),
        ("Error Handling", test_missing_features),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except requests.exceptions.ConnectionError:
            print(f"\nError: Could not connect to API server at {API_BASE_URL}")
            print("Please make sure the server is running.")
            return
        except Exception as e:
            print(f"\nError in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)


if __name__ == "__main__":
    main()

