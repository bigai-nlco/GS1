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


@pytest.mark.parametrize("criteria", [[], ["single"], list(range(11))])
def test_score_limits(criteria):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {
                "model": "GS1",
                "state": "test",
                "questions": {"q": {"type": "score", "instructions": "Rate.", "criteria": criteria}},
            }
        )
