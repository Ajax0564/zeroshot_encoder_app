import asyncio
import time
from ..models import (
    GLiNERModel,
    ClassificationRequest,
    EntityRequest,
)

async def main():
    print("Loading model...")

    model = GLiNERModel()

    print("Model loaded!\n")

    # -------------------------
    # Entity extraction
    # -------------------------

    entity_request = EntityRequest(
        texts="Patient was prescribed Metformin 500mg twice daily for Type 2 Diabetes. "
              "She reported fatigue and occasional dizziness. Liver function tests ordered.",
        labels={
            "drug": "Pharmaceutical drugs, medications, or treatment names",
            "disease": "Medical conditions, illnesses, or disorders",
            "symptom": "Clinical symptoms or patient-reported symptoms",
            "dosage": "Medication amounts like '50mg' or '2 tablets daily'",
            "organ": "Body parts or organs mentioned in medical context",
        },
        task_name="medical_entities",
        include_confidence=True,
        threshold=0.5,
    )

    start = time.perf_counter()

    entity_result = await model.entities(entity_request)

    elapsed = time.perf_counter() - start

    print("========== ENTITY RESULT ==========")
    print(entity_result)
    print(f"\nEntity inference time: {elapsed:.3f}s\n")


    # -------------------------
    # Single-label classification
    # -------------------------

    classification_request = ClassificationRequest(
        texts=[
            "Patient has severe diabetes and requires immediate treatment.",
            "Patient is stable and requires routine follow-up.",
            "Patient reports mild fatigue.",
        ],
        labels=["low", "medium", "high"],
        task_name="severity",
        is_multilabel=False,
        include_confidence=True,
    )

    start = time.perf_counter()

    classification_result = await model.classify(
        classification_request
    )

    elapsed = time.perf_counter() - start

    print("========== CLASSIFICATION RESULT ==========")
    print(classification_result)
    print(f"\nClassification inference time: {elapsed:.3f}s\n")


    # -------------------------
    # Multi-label classification
    # -------------------------

    multilabel_request = ClassificationRequest(
        texts=[
            "Patient has diabetes and cardiovascular disease.",
            "Patient has neurological symptoms and diabetes.",
            "Patient requires oncology and cardiology follow-up.",
        ],
        labels=[
            "diabetes",
            "cardiology",
            "oncology",
            "neurology",
        ],
        task_name="topics",
        is_multilabel=True,
        include_confidence=True,
    )

    start = time.perf_counter()

    multilabel_result = await model.classify(
        multilabel_request
    )

    elapsed = time.perf_counter() - start

    print("========== MULTI-LABEL RESULT ==========")
    print(multilabel_result)
    print(f"\nMultilabel inference time: {elapsed:.3f}s\n")




if __name__ == "__main__":
    asyncio.run(main())
