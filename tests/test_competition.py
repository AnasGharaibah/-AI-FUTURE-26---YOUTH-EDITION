"""Competition pipeline pieces that need no LLM: parsing, snippets, lexicon router, citations, number words."""
import pytest

from Agent import competition, router
from Agent.competition import (cited_numbers, situation_hints, strip_citations, strip_offers, overlap_fallback,
                               wanted_fields)
from Agent.services import SERVICE_IDS, context_text, load, snippets, sources_block, split_text
from DB.arabic import check_numbers, drop_number_sentences, number_words


def test_all_twelve_services_have_the_nine_fields():
    services = load()
    assert set(services) == set(SERVICE_IDS)
    for s in services.values():
        assert len(s["fields"]) == 9, s["id"]
        assert "الرئيسية" not in "".join(s["fields"].values())         # website menu dropped
    assert services["passport_damaged"]["fields"]["الرسوم"] == "100 دينار"
    assert "]" not in services["id_card_lost"]["fields"]["أماكن تقديم الخدمة"]


def test_snippets_are_numbered_and_short():
    snips = snippets(["passport_damaged"], ["الرسوم", "ملاحظات"])
    assert [s["field"] for s in snips] == ["ملاحظات", "الرسوم"]
    block = sources_block(snips)
    assert block.startswith("[1] (جواز سفر بدل تالف | ملاحظات)\n")
    assert "[2] (جواز سفر بدل تالف | الرسوم)\n100 دينار" in block
    assert all(len(s["text"]) <= 600 for s in snippets(list(SERVICE_IDS)))
    assert context_text(snips[1]) == "جواز سفر بدل تالف - الرسوم: 100 دينار"


def test_split_text_cuts_at_line_breaks():
    parts = split_text("\n".join(["x" * 250] * 5), 600)
    assert [len(p) for p in parts] == [501, 501, 250]


@pytest.mark.parametrize("q, services", [
    ("كم يستغرق تجديد دفتر العائلة؟", ["family_book_renewal"]),
    ("وين بقدر أطلع جواز سفر بدل فاقد؟", ["passport_lost"]),
    ("قديش صورة شخصية بحتاج للبطاقة الذكية لأول مرة، وقديش لتجديد جواز السفر؟",
     ["id_card_first_issue", "passport_renewal"]),
    ("عمري 17 وجوازي باقي عليه 3 شهور، كم رسوم التجديد؟", ["passport_renewal"]),
    ("أيهما أغلى: جواز بدل تالف ولا بدل فاقد؟", ["passport_damaged", "passport_lost"]),
    ("الباسبور خلص، شو بعمل؟", ["passport_renewal"]),
    ("الهوية تبعتي انكسرت، قديش بتكلف؟", ["id_card_damaged"]),
    ("ضاع جوازي لأول مرة، كم بدفع؟", ["passport_lost"]),
    ("جوازي ضاع، والهوية كمان", ["passport_lost", "id_card_lost"]),
    ("ابني ضاع جوازه وهو مسافر، وكمان بطاقة زوجتي الذكية انكسرت. كم رسوم بدل جواز ابني (أول مرة يضيع) "
     "وكم رسوم بطاقة زوجتي؟", ["passport_lost", "id_card_damaged"]),
    ("شو بطاقة الجسور المطلوبة لتجديد دفتر العائلة؟", ["family_book_renewal"]),
    ("تزوجت من 4 شهور وما فتحت دفتر عائلة، في غرامة؟", ["family_book_first_issue"]),
    ("How much does it cost to renew an expired passport?", ["passport_renewal"]),
])
def test_lexicon_routes_without_llm(q, services):
    r = router.lexicon(q)
    assert not r.unsure
    assert r.services == services


def test_document_without_situation_means_all_four():
    r = router.lexicon("إذا توفي الأب والأم، مين بصير رب الأسرة؟")
    assert not r.unsure and r.services == [f"family_book_{s}" for s in ("first_issue", "renewal", "lost", "damaged")]


@pytest.mark.parametrize("q, why", [
    ("بدي أجدد، شو المطلوب؟", "no_document"),
    ("كم سعر تذكرة الطيران من عمان إلى دبي؟", "out_of_scope_word"),
    ("كم رسوم تجديد رخصة القيادة؟", "out_of_scope_word"),
])
def test_lexicon_unsure_goes_to_llm(q, why):
    assert router.lexicon(q).unsure == why


def test_clarification_asks_only_the_missing_part():
    assert router.clarification(router.Route(sits=["renewal"])) == "هل تقصد جواز السفر أم دفتر العائلة أم البطاقة الذكية؟"
    assert "دفتر العائلة" in router.clarification(router.Route(docs=["family_book"]))


def test_route_asks_when_the_llm_guesses_every_document(monkeypatch):
    monkeypatch.setattr(router, "llm_route", lambda *a, **k: {
        "in_scope": True, "needs_clarification": False, "standalone_question": "x",
        "services": ["passport_renewal", "id_card_renewal", "family_book_renewal"]})
    r = router.route("بدي أجدد، شو المطلوب؟")
    assert r.needs_clarification and r.services == []


@pytest.mark.parametrize("q, state, services", [
    ("وإذا كان تالف؟", (["passport"], ["renewal"]), ["passport_damaged"]),          # only the situation changes
    ("وللهوية؟", (["family_book"], ["damaged"]), ["id_card_damaged"]),              # only the document changes
    ("للبطاقة الذكية", ([], ["lost"]), ["id_card_lost"]),                            # answer to "which document?"
    ("وإذا صارت تالت مرة؟", (["passport"], ["lost"]), ["passport_lost"]),             # loss count, still lost
    ("وكم بكلف إذا أول مرة؟", (["id_card"], ["lost"]), ["id_card_lost"]),             # "أول مرة" = first loss
    ("ووين بقدم؟", (["family_book"], ["renewal"]), ["family_book_renewal"]),         # nothing named: both kept
])
def test_follow_up_keeps_the_slots_it_does_not_name(monkeypatch, q, state, services):
    monkeypatch.setattr(router, "llm_route", lambda *a, **k: pytest.fail("no LLM call for a follow-up"))
    r = router.route(q, [("سؤال سابق", "جواب")], state)
    assert r.services == services and r.source == "state"


def test_first_message_without_a_document_is_asked_back(monkeypatch):
    monkeypatch.setattr(router, "llm_route", lambda *a, **k: {"in_scope": True, "services": ["passport_first_issue"]})
    r = router.route("وكم بتكلف؟")
    assert r.needs_clarification and r.services == []


def test_citations():
    assert cited_numbers("a [2] b [1, 3] c [2-4] [9] [1][2]", 5) == [2, 1, 3, 4]
    assert strip_citations("الرسوم 50 دينار [1]. والمدة 45 دقيقة [2][3].") == "الرسوم 50 دينار. والمدة 45 دقيقة."


def test_strip_offers_and_greetings():
    assert strip_offers("مرحباً بك! الرسوم 50 دينار [1].\n\nهل ترغب بمعرفة المزيد؟") == "الرسوم 50 دينار [1]."
    assert strip_offers("الرسوم 50 دينار [1]. هل جوازك منتهي؟") == "الرسوم 50 دينار [1]. هل جوازك منتهي؟"


def test_number_words():
    assert number_words("خمسة وعشرون دينارا ودينارين وثلاثة أشهر") == ["25", "2", "3"]
    assert number_words("مئة وخمسة وعشرون، ثلاث مئة") == ["125", "300"]
    assert number_words("لمن أتم سن السادسة عشر") == []                       # ordinal: consumed, not 10
    assert number_words("لمن أتم سن السادسة عشر", ordinals=True) == ["16"]
    assert check_numbers("الرسوم خمسة وعشرون دينارا [1]", "25 دينار") == []
    assert check_numbers("الرسوم 2 دينار", "دينارين لتجديد دفتر العائلة") == []
    assert check_numbers("لمن أتم 16 سنة", "لمن أتم السادسة عشر") == []
    assert check_numbers("الرسوم ستون دينارا", "50 دينار") == ["60"]
    assert drop_number_sentences("الرسوم ستون دينارا. المدة 45 دقيقة.", ["60"]) == "المدة 45 دقيقة."
    # amounts glued to words in the documents ("250دينار", "10دنانير غرامة") are still found
    assert check_numbers("الرسوم 250 دينار للمرة الثانية", "250دينار لاي مرة") == []
    assert check_numbers("غرامة 10 دنانير", "10دنانير غرامة تاخير") == []


def test_situation_hints():
    h = situation_hints("عمري 17 وجوازي باقي عليه 3 شهور، كم رسوم التجديد؟")
    assert "16 سنة أو أكثر وأقل من 18" in h[0] and "أقل من 180 يوماً" in h[1]
    assert "أكثر من 90 يوماً" in situation_hints("تزوجت من 4 شهور")[0]
    assert situation_hints("ضاع جوازي وأنا برا الأردن") == ["المستخدم خارج الأردن."]
    assert situation_hints("كم رسوم الجواز؟") == []
    assert situation_hints("عمري 25 وجوازي انتهى")[1] == "الوثيقة منتهية الصلاحية."
    assert situation_hints("دفتر العائلة غير منتهي") == ["الوثيقة ما زالت سارية (غير منتهية)."]
    assert "المرة الثانية" in situation_hints("ضاع للمرة الثانية بعد سنتين من أول مرة")[0]
    assert situation_hints("بدي أطلع جواز لأول مرة") == []                 # not a loss count


def test_fields_follow_the_question():
    assert wanted_fields("كم رسوم تجديد الجواز؟") == ["الشروط", "ملاحظات", "الرسوم"]
    assert wanted_fields("إذا توفي الأب والأم، مين بصير رب الأسرة؟") is None


def test_overlap_fallback_prefers_matching_numbers():
    snips = snippets(["passport_damaged"])
    best = overlap_fallback("رسوم جواز السفر بدل التالف 100 دينار", snips)
    assert snips[best[0] - 1]["field"] == "الرسوم"


def test_answer_without_citation_retries_then_uses_overlap(monkeypatch):
    calls = []

    def fake(question, standalone, snips, model, reminder="", convo="", hints=None):
        calls.append(reminder)
        return "رسوم جواز السفر بدل تالف 100 دينار."
    monkeypatch.setattr(competition, "_generate", fake)
    dbg = {"flags": []}
    ans, ctx = competition.Chat()._answer("كم رسوم جواز بدل تالف؟", "كم رسوم جواز بدل تالف؟", ["passport_damaged"], dbg)
    assert calls == ["", competition.CITE_REMINDER]                         # one retry with the reminder
    assert ctx == ["جواز سفر بدل تالف - الرسوم: 100 دينار"] and "overlap_fallback" in dbg["flags"]


def test_unsupported_number_is_retried_then_dropped(monkeypatch):
    replies = iter(["الرسوم 150 دينار [2]. المدة 45 دقيقة [1].", "الرسوم 150 دينار [2]. المدة 45 دقيقة [1]."])
    monkeypatch.setattr(competition, "_generate", lambda *a, **k: next(replies))
    monkeypatch.setattr(competition, "build_snippets", lambda s, q: snippets(["passport_damaged"], ["الزمن المستغرق", "الرسوم"]))
    dbg = {"flags": []}
    ans, ctx = competition.Chat()._answer("كم الرسوم والمدة لجواز بدل تالف؟", "x", ["passport_damaged"], dbg)
    assert ans == "المدة 45 دقيقة."
    assert ctx == ["جواز سفر بدل تالف - الزمن المستغرق: 45 دقيقة"]


def test_typos_and_full_pages():
    from Agent.competition import fix_typos
    q = fix_typos("انا بالاردن ةمش عسكري و جةاز سفري مخلص ةرقه")
    assert q == "انا بالاردن ومش عسكري و جواز سفري مخلص ورقه"
    assert fix_typos("الرسوم مدة صلاحية") == "الرسوم مدة صلاحية"              # ة at the end of a word stays
    assert situation_hints(q) == ["صفحات جواز السفر ممتلئة.", "المستخدم ليس من العسكريين أو الأجهزة الأمنية.",
                                  "المستخدم داخل الأردن."]                   # "مخلص ورقه" is not "expired"
    assert wanted_fields("كم المبلغ؟") == ["الشروط", "ملاحظات", "الرسوم"]


def test_one_question_and_dangling_line():
    from Agent.competition import drop_dangling, one_question
    asked = one_question("يختلف المبلغ، فيرجى الإجابة على الأسئلة التالية:\n1. هل أنت داخل الأردن؟\n2. هل أنت عسكري؟")
    assert asked == "يختلف المبلغ، فيرجى الإجابة على السؤال التالي:\nهل أنت داخل الأردن؟"
    assert drop_dangling("تحتاج:\n- البطاقة\n\nفإن المبلغ المطلوب هو:") == "تحتاج:\n- البطاقة"


def test_no_question_back_when_the_document_is_known():
    from Agent.competition import strip_questions
    assert strip_questions("الرسوم 50 دينار [3].\n\nهل جوازك منتهي؟") == "الرسوم 50 دينار [3]."
    assert strip_questions("الرسوم 50 دينار [3]. هل أنت داخل الأردن؟") == "الرسوم 50 دينار [3]."
    assert strip_questions("الرسوم 50 دينار [3].") == "الرسوم 50 دينار [3]."


def test_asked_back_is_retried_then_declined(monkeypatch):
    replies = iter(["هل جوازك منتهي؟", "هل أنت داخل الأردن؟"])
    monkeypatch.setattr(competition, "_generate", lambda *a, **k: next(replies))
    dbg = {"flags": []}
    ans, ctx = competition.Chat()._answer("كم مدة صلاحية الجواز؟", "x", ["passport_renewal"], dbg)
    assert ans == competition.NOT_FOUND and ctx == [] and "retry_asked_back" in dbg["flags"]


def test_only_the_fee_lines_that_apply_are_sent(monkeypatch):
    prompts = []

    def fake_chat(messages, **kw):
        import re
        prompts.append(messages[-1]["content"])
        fee = re.search(r"\[(\d+)\] \(تجديد جواز السفر \| الرسوم\)", prompts[-1]).group(1)
        note = re.search(r"\[(\d+)\] \(تجديد جواز السفر \| ملاحظات\)\n[^\n]*المستعجلة", prompts[-1]).group(1)
        return f"الرسوم 50 دينار [{fee}] ويضاف 25 دينار [{note}]، المجموع 75 دينار."
    monkeypatch.setattr(competition.ollama_http, "chat", fake_chat)
    res = competition.Chat().ask("عمري 40 وجوازي منتهي وبدي أجدده بمكتب الخدمة المستعجلة، قديش المجموع؟")
    sources = prompts[0].split("سؤال المستخدم")[0]
    assert "فوق سن السادسة عشر" in sources and "دون السادسة عشر" not in sources and "غير المنتهي" not in sources
    assert "مكتب الخدمة المستعجلة" in sources                                   # notes always kept
    assert "75" in res["answer"]                                                  # a sum of the selected amounts
    assert [c.split(":")[0] for c in res["contexts"]] == ["تجديد جواز السفر - الرسوم", "تجديد جواز السفر - ملاحظات"]


def test_long_story_is_split_per_service(monkeypatch):
    prompts = []

    def fake_chat(messages, **kw):
        prompts.append(messages[-1]["content"])
        return "الرسوم [1]."
    monkeypatch.setattr(competition.ollama_http, "chat", fake_chat)
    res = competition.Chat().ask("مرحبا، ابني عمره 10 سنين وبدنا نطلعله جواز سفر لأول مرة عشان نسافر. "
                                 "وأخوي ساكن بألمانيا وجوازه ضايل عليه شهرين وبده يجدده من السفارة. "
                                 "قديش رح ندفع على كل وحدة؟")
    assert res["debug"]["route"] == ["passport_first_issue", "passport_renewal"]
    assert len(prompts) == 2 and "story:2" in res["debug"]["flags"]
    son, brother = prompts
    assert "دون سن 16" in son and "داخل البلاد" not in son and "خارج البلاد" not in son.split("سؤال المستخدم")[0]
    assert "خارج البلاد" in brother and "ألمانيا" in brother and "عمره 10" not in brother
    assert res["answer"].startswith("• جواز سفر لأول مرة:")


def test_chat_remembers_the_users_situation(monkeypatch):
    """The user answers the bot's question in a later message: the answer prompt sees the earlier messages and the
    facts computed from the whole chat (newest first)."""
    prompts = []

    def fake_chat(messages, **kw):
        prompts.append(messages[-1]["content"])
        return "الرسوم 20 دينار [1]."
    monkeypatch.setattr(competition.ollama_http, "chat", fake_chat)
    monkeypatch.setattr(router, "llm_route", lambda *a, **k: {})
    bot = competition.Chat()
    bot.ask("كم رسوم تجديد الجواز؟")
    bot.ask("انا بالاردن ةمش عسكري وجةاز سفري مخلص ةرقه")
    res = bot.ask("كم المبلغ؟")
    assert res["debug"]["route"] == ["passport_renewal"]
    assert "- انا بالاردن ومش عسكري وجواز سفري مخلص ورقه" in prompts[-1]
    assert "صفحات جواز السفر ممتلئة." in prompts[-1] and "المستخدم داخل الأردن." in prompts[-1]
    assert "(جواز سفر بدل تالف" not in prompts[-1] and "| الرسوم)" in prompts[-1]
