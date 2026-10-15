"""Fee / note line selection (Agent/fees.py): every rule with two phrasings that are NOT in test_questions.json.

Each case: (service, message, phrases the selected fee lines must contain - one per selected line, or None when the
rule must stay undecided and keep all lines, phrases of note lines that must be pointed at)."""
import json
from pathlib import Path

import pytest

from Agent import fees
from Agent.competition import chat_facts
from Agent.services import load
from DB.arabic import normalize_search

CASES = [
    # passport renewal
    ("passport_renewal", "عمري 30 وجوازي منتهي من شهر، كم بدفع للتجديد؟", ["فوق سن السادسة عشر"], []),
    ("passport_renewal", "انتهت صلاحية باسبوري وأنا عمري 45، شو رسوم تجديده", ["فوق سن السادسة عشر"], []),
    ("passport_renewal", "جوازي ساري لسنة كمان بس بدي أجدده، قديش؟", ["غير المنتهي"], []),
    ("passport_renewal", "باقي على جوازي سنتين وبدي أجدده هلأ", ["غير المنتهي"], []),
    ("passport_renewal", "ابني عمره 12 وجوازه انتهى، كم تجديده؟", ["دون السادسة عشر"], []),
    ("passport_renewal", "بنتي عمرها 9 سنين وبدها تجدد جوازها", ["دون السادسة عشر"], []),
    ("passport_renewal", "جوازي مليان صفحات وما انتهى، كم التجديد؟", ["الممتلئ الصفحات"], []),
    ("passport_renewal", "صفحات جوازي امتلأت وهو لسا ساري", ["الممتلئ الصفحات"], []),
    ("passport_renewal", "أنا مغترب بقطر وبدي أجدد جوازي", ["خارج البلاد"], []),
    ("passport_renewal", "بجدد الجواز من السفارة بألمانيا، قديش؟", ["خارج البلاد"], []),
    ("passport_renewal", "أنا عسكري بالجيش وبدي أجدد جوازي", ["لأفراد القوات"], []),
    ("passport_renewal", "أنا بالدرك وجوازي منتهي، كم الرسوم؟", ["لأفراد القوات"], []),
    ("passport_renewal", "جوازي منتهي وعمري 33 وبدي أقدم مستعجل", ["فوق سن السادسة عشر"], ["المستعجلة"]),
    ("passport_renewal", "بدي أجدد بمكتب الخدمة المستعجلة وجوازي خلص وعمري 50", ["فوق سن السادسة عشر"], ["المستعجلة"]),
    ("passport_renewal", "قديش رسوم تجديد الباسبور؟", None, []),
    ("passport_renewal", "كم بكلف تجديد جواز السفر؟", None, []),
    # passport first issue
    ("passport_first_issue", "بنتي عمرها 7 بدي أطلعلها جواز", ["دون سن 16"], []),
    ("passport_first_issue", "ابني 13 سنة وبدنا أول جواز إله", ["دون سن 16"], []),
    ("passport_first_issue", "عمري 22 وبدي أطلع جواز لأول مرة وأنا بالأردن", ["داخل البلاد"], []),
    ("passport_first_issue", "أنا بالغ وساكن بالأردن، بدي أول جواز", ["داخل البلاد"], []),
    ("passport_first_issue", "عمري 30 ومقيم بالإمارات، أول مرة بطلع جواز", ["خارج البلاد"], []),
    ("passport_first_issue", "بدي أطلع جواز لأول مرة من القنصلية بتركيا", ["خارج البلاد"], []),
    # passport lost
    ("passport_lost", "ضاع باسبوري أول مرة بحياتي", ["للمرة الاولى"], []),
    ("passport_lost", "انسرق جوازي وهاي أول مرة", ["للمرة الاولى"], []),
    ("passport_lost", "ضاع جوازي مرة ثانية بعد 3 سنين", ["لاي مرة"], []),
    ("passport_lost", "هاي ثالث مرة بضيع فيها الجواز", ["لاي مرة"], []),
    # family book first issue (late fine after 90 days of marriage)
    ("family_book_first_issue", "كتبنا الكتاب من 6 شهور وما طلعنا دفتر", ["رسوم إصدار الدفتر", "غرامة"], []),
    ("family_book_first_issue", "تزوجنا من سنة ولسا ما فتحنا دفتر عائلة", ["رسوم إصدار الدفتر", "غرامة"], []),
    ("family_book_first_issue", "تزوجت من شهر وبدي أطلع دفتر عائلة", ["رسوم إصدار الدفتر"], []),
    ("family_book_first_issue", "عقد زواجنا من 20 يوم، كم رسوم الدفتر؟", ["رسوم إصدار الدفتر"], []),
    # family book renewal
    ("family_book_renewal", "دفتر العيلة منتهي، كم تجديده؟", ["منتهي المدة"], []),
    ("family_book_renewal", "انتهت صلاحية دفتر العائلة وبدي أجدده", ["منتهي المدة"], []),
    ("family_book_renewal", "دفتر العائلة لسا ساري وبدي أجدده", ["غير منتهي المدة"], []),
    ("family_book_renewal", "دفترنا مش منتهي بس بدنا نجدده", ["غير منتهي المدة"], []),
    # family book lost (pledge / guarantee by loss count)
    ("family_book_lost", "أول مرة بيضيع مني دفتر العائلة", ["5 دنانير", "تعهد خطي"], ["لأول مرة"]),
    ("family_book_lost", "فقدنا دفتر العيلة للمرة الأولى", ["5 دنانير", "تعهد خطي"], ["لأول مرة"]),
    ("family_book_lost", "ضاع الدفتر ثاني مرة خلال سنة", ["5 دنانير", "تعهد خطي", "استيفاء"], ["للمرة الثانية"]),
    ("family_book_lost", "تاني مرة بضيع دفتر العائلة وآخر مرة كانت قبل سنتين",
     ["5 دنانير", "تعهد خطي", "استيفاء"], ["للمرة الثانية"]),
    ("family_book_lost", "ثالث مرة بيضيع دفتر العيلة", ["5 دنانير", "استيفاء"], ["للمرة الثالثة"]),
    ("family_book_lost", "فقدت دفتر العائلة للمرة الثالثة", ["5 دنانير", "استيفاء"], ["للمرة الثالثة"]),
    # family book damaged (previous pledge when the last book was issued < 3 years ago)
    ("family_book_damaged", "تلف دفتر العائلة وطلعناه قبل سنة", ["3 دنانير", "استيفاء"], []),
    ("family_book_damaged", "دفتر العيلة تمزق وآخر إصدار كان من سنتين", ["3 دنانير", "استيفاء"], []),
    ("family_book_damaged", "تلف الدفتر وآخر مرة طلعناه قبل 5 سنوات", ["3 دنانير"], []),
    ("family_book_damaged", "دفتر العائلة خربان وطلعته قبل 4 سنين", ["3 دنانير"], []),
    # id card first issue (late fine from 18)
    ("id_card_first_issue", "عمري 20 وما عمري طلعت هوية", ["2 دينار", "غرامة"], []),
    ("id_card_first_issue", "صار عمري 25 وبدي أطلع بطاقة ذكية أول مرة", ["2 دينار", "غرامة"], []),
    ("id_card_first_issue", "ابني عمره 16 وبده هوية لأول مرة", ["2 دينار"], []),
    ("id_card_first_issue", "بنتي 17 سنة بدها تطلع البطاقة الذكية", ["2 دينار"], []),
    # id card lost (pledge / committee by loss count, two weeks after the police report)
    ("id_card_lost", "أول مرة بتضيع هويتي", ["5 دنانير"], []),
    ("id_card_lost", "ضاعت بطاقتي الذكية لأول مرة", ["5 دنانير"], []),
    ("id_card_lost", "ضاعت هويتي مرة ثانية بعد سنة من آخر مرة", ["5 دنانير", "استيفاء"], ["للمرة الثانية"]),
    ("id_card_lost", "تاني مرة بتضيع بطاقتي الذكية خلال سنتين", ["5 دنانير", "استيفاء"], ["للمرة الثانية"]),
    ("id_card_lost", "ثالث مرة بضيع الهوية", ["5 دنانير", "استيفاء"], ["للمرة الثالثة"]),
    ("id_card_lost", "فقدت البطاقة الشخصية للمرة الثالثة", ["5 دنانير", "استيفاء"], ["للمرة الثالثة"]),
    ("id_card_lost", "ضاعت هويتي، متى بستلم البدل؟", None, ["أسبوعين"]),
    ("id_card_lost", "إذا انسرقت البطاقة الذكية بقدر آخذ البدل بنفس اليوم؟", None, ["أسبوعين"]),
    # notes only
    ("passport_damaged", "تلف جوازي عن قصد شو بصير؟", None, ["متعمد"]),
    ("passport_damaged", "إذا مزقت جوازي متعمد شو العقوبة", None, ["متعمد"]),
    ("family_book_renewal", "توفي أبوي وأمي، مين بصير رب الأسرة؟", None, ["وفاة الاب والام"]),
    ("family_book_first_issue", "أبوي متوفي وكان متزوج زوجتين، كيف الدفتر؟",
     ["رسوم إصدار الدفتر", "غرامة"], ["أكثر من زوجة"]),
]


def _selected(sid: str, q: str):
    f = chat_facts([q], sid.endswith("_lost"))
    p = fees.pick(sid, f, q)
    fields = load()[sid]["fields"]
    fee_lines = fields["الرسوم"].splitlines()
    notes = fields.get("ملاحظات", "").splitlines()
    sel = None if p.fee is None else [line for i, line in enumerate(fee_lines)
                                     if any(fees.matches(k, line, i) for k in p.fee)]
    hit = [line for i, line in enumerate(notes) if any(fees.matches(k, line, i) for k in p.notes)]
    return sel, hit


@pytest.mark.parametrize("sid, q, fee, notes", CASES, ids=[c[1][:30] for c in CASES])
def test_fee_rule(sid, q, fee, notes):
    sel, hit = _selected(sid, q)
    if fee is None:
        assert sel is None, f"should stay undecided, selected {sel}"
    else:
        assert sel is not None and len(sel) == len(fee), sel
        for phrase, line in zip(fee, sel):
            assert normalize_search(phrase) in normalize_search(line), (phrase, line)
    for phrase in notes:
        assert any(normalize_search(phrase) in normalize_search(line) for line in hit), (phrase, hit)


def test_every_rule_has_two_phrasings():
    from collections import Counter
    counts = Counter(sid for sid, *_ in CASES)
    for sid in fees.RULES:
        if fees.RULES[sid] is not fees._keep_all:
            assert counts[sid] >= 2, sid


def test_phrasings_are_not_from_the_test_questions():
    path = Path(__file__).resolve().parent.parent / "test_questions.json"
    if not path.exists():
        pytest.skip("test_questions.json not present")
    asked = {normalize_search(m["question"]) for c in json.loads(path.read_text(encoding="utf-8"))["chats"]
             for m in c["messages"]}
    assert not [q for _, q, *_ in CASES if normalize_search(q) in asked]


def test_amounts_are_read_from_the_lines():
    assert fees.amounts(["50 دينار لمن هم فوق سن السادسة عشر", "يضاف 25 دينار في حال", "تعهد خطي بقيمة (10) دنانير"]) \
        == [50, 25, 10]
