import pytest

from small_talk import (
    CRISIS_RESPONSE,
    EMPATHETIC_RESPONSE,
    classify_chat_intent,
    get_emotional_category,
    get_small_talk_response,
    is_contextual_follow_up,
    is_crisis_message,
)


@pytest.mark.parametrize("question", ["Hi", " HELLO! ", "good morning"])
def test_small_talk_recognizes_greetings(question):
    assert get_small_talk_response(question)


@pytest.mark.parametrize("question", ["How do I reset my password?", "Tell me about billing"])
def test_small_talk_does_not_capture_faq_questions(question):
    assert get_small_talk_response(question) is None


def test_contextual_follow_up_detection():
    assert is_contextual_follow_up("I still have not received it")
    assert not is_contextual_follow_up("How can I update my profile?")


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("I feel tired", "tired"),
        ("I'm stressed", "stressed"),
        ("I feel sad today", "sad"),
        ("I'm bored", "bored"),
        ("I feel overwhelmed", "overwhelmed"),
    ],
)
def test_emotional_support_categories(message, category):
    assert get_emotional_category(message) == category
    assert classify_chat_intent(message) == "emotional"
    assert get_small_talk_response(message) == EMPATHETIC_RESPONSE


def test_emotional_message_with_faq_question_routes_to_search():
    message = "I'm stressed, how can I reduce exam stress?"

    assert get_emotional_category(message) is None
    assert classify_chat_intent(message) == "faq"
    assert get_small_talk_response(message) is None


def test_existing_tired_and_singing_small_talk_response_is_preserved():
    answer = get_small_talk_response("I'm tired but feel like singing")

    assert answer is not None
    assert "gentle song" in answer
    assert classify_chat_intent("I'm tired but feel like singing") == "small_talk"


@pytest.mark.parametrize(
    "message",
    [
        "I might hurt myself",
        "I'm thinking about suicide",
        "I'm in crisis",
        "I want to end my life",
    ],
)
def test_crisis_messages_route_away_from_faq_and_llm(message):
    assert is_crisis_message(message)
    assert classify_chat_intent(message) == "crisis"
    assert CRISIS_RESPONSE


def test_crisis_detection_takes_priority_over_faq_question():
    assert classify_chat_intent(
        "I'm suicidal, can you help me with my account?"
    ) == "crisis"
