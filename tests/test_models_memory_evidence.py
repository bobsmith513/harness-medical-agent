"""记忆与证据模型测试：审核转正闭环 + 证据包语义。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from factories import make_evidence, make_evidence_pack
from harness_agent.models.evidence import (
    Evidence,
    ImageRef,
    RetrievalCandidate,
    SourceRef,
)
from harness_agent.models.memory import AllergyRecord, Memory


class TestMemoryReviewLoop:
    """未审核记忆仅作会话内指针，审核通过才可召回。"""

    def _memory(self, **overrides: object) -> Memory:
        base: dict = {
            "patient_id": "pat-001",
            "content": "患者 2026-03 曾因青霉素过敏就诊。",
            "provenance": "model_inference",
            "confidence": "medium",
            "source_turn": 3,
        }
        base.update(overrides)
        return Memory(**base)  # type: ignore[arg-type]

    def test_session_pointer_by_default(self):
        memory = self._memory()
        assert memory.status == "session_pointer"
        assert memory.can_be_recalled() is False

    def test_pending_review_not_recallable(self):
        memory = self._memory(status="pending_review")
        assert memory.can_be_recalled() is False

    def test_approved_is_recallable_with_review_info(self):
        memory = self._memory(
            status="approved",
            reviewed_at="2026-08-27T10:00:00+00:00",
            reviewer="dr-wang",
        )
        assert memory.can_be_recalled() is True
        assert memory.reviewer == "dr-wang"

    def test_approved_without_review_info_rejected(self):
        with pytest.raises(ValidationError, match="reviewed_at"):
            self._memory(status="approved")

    def test_rejected_without_review_info_rejected(self):
        with pytest.raises(ValidationError, match="reviewer"):
            self._memory(status="rejected", reviewed_at="2026-08-27T10:00:00+00:00")

    def test_unknown_provenance_rejected(self):
        with pytest.raises(ValidationError):
            self._memory(provenance="rumor")


class TestAllergyHardRule:
    """硬规则载体：归一化药名 + ATC 交叉反应。"""

    def test_allergy_record_fields(self):
        record = AllergyRecord(
            patient_id="pat-001",
            drug_name_raw="盘尼西林",
            normalized_drug="penicillin",
            atc_code="J01CE",
            cross_reactants=["amoxicillin", "cephalexin"],
        )
        assert record.atc_code == "J01CE"
        assert record.cross_reactants == ["amoxicillin", "cephalexin"]


class TestEvidenceModels:
    """证据模型：溯源、结构补全区分、证据包复核。"""

    def test_evidence_auto_id_generated(self):
        ev = make_evidence()
        assert ev.evidence_id.startswith("ev-")

    def test_structural_completion_default_false(self):
        assert make_evidence().is_structural_completion is False

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                content="内容",
                source=SourceRef(source_id="src-1", source_type="document"),
                confidence="certain",
                provenance="knowledge_base",
            )

    def test_image_ref_invalid_caption_layer_rejected(self):
        with pytest.raises(ValidationError):
            ImageRef(image_id="img-1", doc_id="doc-1", caption_layer="guess")

    def test_image_ref_with_region(self):
        ref = ImageRef(
            image_id="img-1",
            doc_id="doc-1",
            region=(0.1, 0.2, 0.8, 0.9),
            caption_layer="observation",
        )
        assert ref.region is not None

    def test_high_confidence_evidence_excludes_structural(self):
        pack = make_evidence_pack()
        pack.evidence.append(make_evidence(evidence_id="ev-2", structural=True))
        high = pack.high_confidence_evidence
        assert [e.evidence_id for e in high] == ["ev-1"]

    def test_pack_is_reviewed_semantics(self):
        assert make_evidence_pack(reviewed=True).is_reviewed is True
        assert make_evidence_pack(reviewed=False).is_reviewed is False

    def test_pack_with_blocked_drugs(self):
        pack = make_evidence_pack()
        pack.blocked_drugs = ["penicillin"]
        assert pack.blocked_drugs == ["penicillin"]

    def test_retrieval_candidate_defaults(self):
        candidate = RetrievalCandidate(chunk_id="chunk-1", content="内容")
        assert candidate.dense_rank is None
        assert candidate.sparse_rank is None
        assert candidate.fused_rank is None
        assert candidate.score == 0.0
