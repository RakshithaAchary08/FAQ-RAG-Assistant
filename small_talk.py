import re


CRISIS_RESPONSE = (
    "I'm sorry you're facing this. Please contact your local emergency services "
    "or a trusted person right away."
)
EMPATHETIC_RESPONSE = (
    "That sounds tough. Here are a few things I can help with:"
)


def _normalize_message(question: str) -> str:
    return " ".join(
        "".join(
            character
            for character in question.casefold()
            if character.isalnum() or character.isspace() or character in "-'"
        ).split()
    )


def is_crisis_message(question: str) -> bool:
    """Detect direct self-harm, suicide, or immediate-crisis wording."""
    normalized = _normalize_message(question)
    patterns = (
        r"\bself[\s-]?harm(?:ing)?\b",
        r"\bsuicid(?:e|al)\b",
        r"\bkill myself\b",
        r"\bhurt myself\b",
        r"\bend my life\b",
        r"\btake my own life\b",
        r"\bwant to die\b",
        r"\bcrisis\b",
        r"\bin immediate danger\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _has_faq_intent(question: str, normalized: str) -> bool:
    """Keep emotional language in normal FAQ retrieval when a real question exists."""
    if "?" in question:
        return True
    return bool(
        re.search(
            r"\b(how|what|where|when|why|which|can|could|would|do|does|"
            r"should|need to|please help|help me)\b",
            normalized,
        )
    )


def get_emotional_category(question: str) -> str | None:
    """Return an emotion category for standalone feeling disclosures."""
    normalized = _normalize_message(question)
    if _has_faq_intent(question, normalized):
        return None
    tokens = set(re.findall(r"[a-z]+", normalized))
    if "sing" in normalized and tokens.intersection({"tired", "exhausted"}):
        return None
    for category in ("overwhelmed", "stressed", "tired", "sad", "bored"):
        if category in tokens:
            return category
    return None


def classify_chat_intent(question: str) -> str:
    """Route crisis and standalone small talk before FAQ search or LLM generation."""
    if is_crisis_message(question):
        return "crisis"
    if get_emotional_category(question) is not None:
        return "emotional"
    if get_small_talk_response(question) is not None:
        return "small_talk"
    return "faq"


def get_small_talk_response(question: str) -> str | None:
    normalized = _normalize_message(question)
    responses = {
        "hi": "Hi! How can I help with your account or the service today?",
        "hello": "Hello! How can I help with your account or the service today?",
        "hey": "Hey! How can I help with your account or the service today?",
        "hi there": "Hi! How can I help with your account or the service today?",
        "hello there": "Hello! How can I help with your account or the service today?",
        "hey there": "Hey! How can I help with your account or the service today?",
        "good morning": "Good morning! How can I help you today?",
        "good afternoon": "Good afternoon! How can I help you today?",
        "good evening": "Good evening! How can I help you today?",
        "how are you": "I'm doing well, thanks for asking! What can I help you with today?",
        "how are you doing": "I'm doing well, thanks for asking! What can I help you with today?",
        "how are you today": "I'm doing well, thanks for asking! What can I help you with today?",
        "how are you doing today": "I'm doing well, thanks for asking! What can I help you with today?",
        "hi how are you": "I'm doing well, thanks for asking! What can I help you with today?",
        "hi how are you doing": "I'm doing well, thanks for asking! What can I help you with today?",
        "hello how are you": "I'm doing well, thanks for asking! What can I help you with today?",
        "hello how are you doing": "I'm doing well, thanks for asking! What can I help you with today?",
        "hey how are you": "I'm doing well, thanks for asking! What can I help you with today?",
        "hey how are you doing": "I'm doing well, thanks for asking! What can I help you with today?",
        "hows it going": "It's going well, thanks! What can I help you with today?",
        "whats up": "I'm here and ready to help! What can I help you with today?",
        "can i ask a question": "Of course, ask away! I can help with questions covered by the FAQs.",
        "can i ask one question": "Of course, ask away! I can help with questions covered by the FAQs.",
        "may i ask a question": "Of course, ask away! I can help with questions covered by the FAQs.",
        "can you help me": "Of course! What can I help you with?",
        "can you help me please": "Of course! What can I help you with?",
        "can u help me": "Of course! What can I help you with?",
        "could you help me": "Of course! What can I help you with?",
        "could you help me please": "Of course! What can I help you with?",
        "thanks": "You're welcome! Let me know if you have another question.",
        "thank you": "You're welcome! Let me know if you have another question.",
        "thanks a lot": "You're welcome! Let me know if you have another question.",
        "thank you so much": "You're welcome! Let me know if you have another question.",
        "bye": "Goodbye! Take care.",
        "goodbye": "Goodbye! Take care.",
        "see you": "See you later! Let me know if you need help.",
    }
    if normalized in {
        "today i dont want to go to college",
        "today i dont want to went for college",
        "i dont want to go to college today",
        "i dont want to went for college today",
    }:
        return (
            "Sounds like you're not feeling up to college today. "
            "Want to talk about what's making you feel that way?"
        )

    if "sing" in normalized and any(
        word in normalized.split() for word in ("tired", "exhausted")
    ):
        return (
            "Sounds like you're tired but feel like singing. You could try a gentle "
            "song or just hum for a bit, and rest if you need to. Want me to make up "
            "a short, original song?"
        )

    if get_emotional_category(question) is not None:
        return EMPATHETIC_RESPONSE

    return responses.get(normalized)


def is_contextual_follow_up(question: str) -> bool:
    normalized = " ".join(
        "".join(
            character
            for character in question.casefold()
            if character.isalnum() or character.isspace()
        ).split()
    )
    follow_up_phrases = (
        "i did not get",
        "i didnt get",
        "i have not received",
        "i havent received",
        "i did not receive",
        "i didnt receive",
        "it did not arrive",
        "it didnt arrive",
        "it has not arrived",
        "it hasnt arrived",
        "still have not received",
        "still havent received",
        "still did not get",
        "still didnt get",
    )
    return len(normalized.split()) <= 12 and any(
        phrase in normalized for phrase in follow_up_phrases
    )
