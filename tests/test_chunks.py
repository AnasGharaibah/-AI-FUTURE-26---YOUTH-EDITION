from DB.chunks import Chunker, arabic_heading, ntok

DOC = """# دليل خدمة تجديد الجواز

الوثائق المطلوبة:

1. جواز السفر القديم
2. صورة شخصية حديثة
3. إيصال دفع الرسوم

الرسوم: 25 دينارا للتجديد العادي.

المادة 7
يجب على المواطن تقديم الطلب خلال ثلاثين يوما من تاريخ الانتهاء. ويحق للمركز رفض الطلب غير المكتمل؟

أولاً: الشروط
أن يكون المتقدم أردني الجنسية.
"""


def test_headings():
    assert arabic_heading("المادة (7)")[:2] == (3, "المادة 7")
    assert arabic_heading("المادة 5: التزامات المستفيد")[1] == "المادة 5 - التزامات المستفيد"
    assert arabic_heading("الفصل الثاني")[0] == 2
    assert arabic_heading("ثانياً: الخطوات")[0] == 5
    assert arabic_heading("الوثائق المطلوبة:") == (6, "الوثائق المطلوبة", "")
    assert arabic_heading("الرسوم: 25 دينارا") == (6, "الرسوم", "25 دينارا")
    assert arabic_heading("هذه جملة عادية طويلة عن الخدمة") is None
    assert arabic_heading("| الرسوم: | 5 |") is None


def test_ntok_arabic_heavier():
    assert ntok("كلمة " * 10) > ntok("word " * 10)


def test_sections_and_lists_stay_whole():
    chunks, parents = Chunker().chunk_text(DOC, "guide.docx")
    paths = {c["section"] for c in chunks}
    assert any(p.endswith("الوثائق المطلوبة") for p in paths)
    docs = next(c for c in chunks if c["section"].endswith("الوثائق المطلوبة"))
    assert all(x in docs["text"] for x in ("1. جواز", "2. صورة", "3. إيصال"))
    assert any(c["section"].endswith("المادة 7") for c in chunks)
    assert any("25 دينارا" in c["text"] and c["section"].endswith("الرسوم") for c in chunks)


def test_long_list_split_repeats_heading():
    steps = "\n".join(f"{i}. الخطوة رقم {i} ثم تابع الإجراء كما هو موضح في الدليل الرسمي للخدمة" for i in range(1, 60))
    chunks, _ = Chunker().chunk_text(f"الخطوات:\n\n{steps}\n", "g.docx")
    assert len(chunks) > 1
    assert all(c["text"].startswith("الخطوات:") for c in chunks[1:])


def test_sentence_split_arabic_punctuation():
    from DB.chunks import SENT_RE
    assert len([s for s in SENT_RE.split("هل يجوز؟ نعم، يجوز؛ بشرط ذلك. تم") if s]) >= 4
