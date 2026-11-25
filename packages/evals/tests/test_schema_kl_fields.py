"""KL-2/KL-3 optional Answer fields: v0 answers keep validating, defaults are inert."""

from rav_evals.schema import Answer


def test_v0_answer_validates_with_kl_defaults():
    a = Answer.model_validate(
        {
            "claims": [
                {"type": "quote", "text": "t", "citations": [{"ref": "r", "quoted_text": "q"}]}
            ]
        }
    )
    c = a.claims[0]
    assert a.scope is None and a.unchecked_core_books == []
    assert c.kl_item_ids == [] and c.relations == [] and c.flags == []
    assert c.citations[0].source_role == "primary" and c.citations[0].witness_id is None


def test_kl_fields_round_trip():
    raw = {
        "claims": [
            {
                "type": "interpretation",
                "text": "t",
                "kl_item_ids": [3],
                "flags": ["unusual_interpretation", "lens"],
                "relations": [
                    {
                        "table": "edges",
                        "id": 1,
                        "rel": "x",
                        "evidence_origin": "sefaria",
                        "candidate": True,
                        "evidence_refs": ["a"],
                    }
                ],
                "citations": [
                    {
                        "ref": "r",
                        "quoted_text": "q",
                        "source_role": "secondary_lens",
                        "witness_id": 2,
                    }
                ],
            }
        ],
        "scope": "partial",
        "unchecked_core_books": ["B"],
    }
    assert Answer.model_validate(raw).model_dump(mode="json")["claims"][0]["relations"][0][
        "candidate"
    ]
