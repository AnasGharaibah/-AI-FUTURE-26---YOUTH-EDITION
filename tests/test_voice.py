from Agent.voice import clean_text, digits_to_words, number_to_arabic_words as w


def test_numbers():
    assert w(0) == "صفر"
    assert w(5) == "خمسة"
    assert w(12) == "اثنا عشر"
    assert w(25) == "خمسة وعشرون"
    assert w(100) == "مئة"
    assert w(150) == "مئة وخمسون"
    assert w(200) == "مئتان"
    assert w(999) == "تسعمئة وتسعة وتسعون"
    assert w(1000) == "ألف"
    assert w(2000) == "ألفان"
    assert w(3000) == "ثلاثة آلاف"
    assert w(12000) == "اثنا عشر ألف"
    assert w(1250) == "ألف ومئتان وخمسون"
    assert w(999999).startswith("تسعمئة وتسعة وتسعون ألف")
    assert w(1_000_000) == "1 0 0 0 0 0 0"        # out of range -> digits


def test_digits_to_words():
    assert digits_to_words("الرسوم 25 دينارا") == "الرسوم خمسة وعشرون دينارا"
    assert digits_to_words("2.5") == "اثنان فاصلة خمسة"
    assert digits_to_words("0791234567") == " ".join(["صفر", "سبعة", "تسعة", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة"])


def test_clean_text_arabic_speakable():
    out = clean_text("الرسوم 25 دينارا [1][2]. **ملاحظة** يرجى المراجعة. ثالثا. رابعا.")
    assert "[" not in out and "*" not in out and "25" not in out and "خمسة وعشرون" in out
    assert len([x for x in out.split(".") if x.strip()]) <= 3
