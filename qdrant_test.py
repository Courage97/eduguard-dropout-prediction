import random
import json
import urllib.request

# Generate a realistic random 768-dimensional vector
random.seed(42)  # reproducible, so we know exactly what we inserted
vector = [round(random.uniform(-1, 1), 6) for _ in range(768)]

payload = {
    "points": [
        {
            "id": 1,
            "vector": vector,
            "payload": {
                "hospital_id": "test-hospital-1",
                "patient_id": "test-patient-1",
                "record_id": "test-record-1",
                "section_type": "presenting_complaint",
                "record_version": 1,
                "synced_at": "2026-09-06T10:00:00Z"
            }
        }
    ]
}

req = urllib.request.Request(
    "http://localhost:6333/collections/clinical_sections/points",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="PUT"
)

with urllib.request.urlopen(req) as response:
    print("INSERT RESULT:")
    print(response.read().decode())

# Now query it back using the same vector - should return itself as the top match
query_payload = {
    "vector": vector,
    "limit": 3,
    "with_payload": True
}

req2 = urllib.request.Request(
    "http://localhost:6333/collections/clinical_sections/points/search",
    data=json.dumps(query_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req2) as response:
    print("\nSEARCH RESULT:")
    print(json.dumps(json.loads(response.read().decode()), indent=2))