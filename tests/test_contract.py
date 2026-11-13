import pytest
from pydantic import ValidationError

from gs1.model.schema import Choice, DecisionRequest, Score, make_answer


@pytest.mark.parametrize("p", [[0.6, 0.3, 0.1], [1.0], [0.5, 0.5]])
def test_choice_returns_complete_distribution(p):
    q = Choice(type="choice", instructions="Choose.", criteria={str(i): str(i) for i in range(len(p))})
    answer = make_answer(q, p)
    assert answer.choice == "0"
    assert list(answer.probabilities.values()) == p
    assert 0 <= answer.confidence <= 1


def test_score_uses_zero_based_expectation():
    answer = make_answer(
        Score(type="score", instructions="Rate.", criteria=["Low", "Medium", "High"]), [0.25, 0.5, 0.25]
    )
    assert answer.score == 1
    assert answer.confidence == pytest.approx(0.25)


@pytest.mark.parametrize("criteria", [[], ["single"], ["level"] * 257])
def test_score_limits(criteria):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {
                "model": "GS1",
                "state": "test",
                "questions": {"q": {"type": "score", "instructions": "Rate.", "criteria": criteria}},
            }
        )


@pytest.mark.parametrize(
    "kind,count",
    [
        ("choice", 1),
        ("choice", 255),
        ("choice", 256),
        ("score", 2),
        ("score", 11),
        ("score", 255),
        ("score", 256),
    ],
)
def test_candidate_counts_within_limits(kind, count):
    criteria = {str(i): str(i) for i in range(count)} if kind == "choice" else [str(i) for i in range(count)]
    request = DecisionRequest.model_validate(
        {
            "model": "GS1",
            "state": "test",
            "questions": {"q": {"type": kind, "instructions": "Choose a level.", "criteria": criteria}},
        }
    )
    assert len(request.questions["q"].criteria) == count


@pytest.mark.parametrize("count", [0, 257])
def test_choice_limits(count):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {
                "model": "GS1",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose.",
                        "criteria": {str(i): str(i) for i in range(count)},
                    }
                },
            }
        )


def test_score_at_maximum_level():
    question = Score(instructions="Rate.", criteria=[str(i) for i in range(256)])
    answer = make_answer(question, [0.0] * 255 + [1.0])
    assert answer.score == 255
    assert len(answer.probabilities) == len(answer.legend) == 256
    assert answer.probabilities["255"] == 1
    assert answer.legend["255"] == "255"
