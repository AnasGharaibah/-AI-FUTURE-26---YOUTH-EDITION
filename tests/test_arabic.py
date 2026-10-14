from DB.arabic import is_arabic, looks_reversed, normalize_display, normalize_search, tokens


def test_display_presentation_forms():
    assert normalize_display("ﻳﺴﻤﺢ ﻟﻠﻤﻮﻇﻔﻴﻦ") == "يسمح للموظفين"


def test_display_tatweel_digits_spaces():
    assert normalize_display("الرســـوم:  ٢٥  دينار ۱۰") == "الرسوم: 25 دينار 10"
    assert normalize_display("") == ""


def test_display_keeps_newlines_and_hamza():
    out = normalize_display("أولاً:  الشروط\n  ثانياً: الخطوات")
    assert out == "أولاً: الشروط\n  ثانياً: الخطوات"      # indentation kept


def test_search_unifies_letters_and_drops_diacritics():
    assert normalize_search("إِصْدَارُ أَوْرَاقٍ") == "اصدار اوراق"
    assert normalize_search("مدرسة") == normalize_search("مدرسه")
    assert normalize_search("على") == normalize_search("علي")
    assert normalize_search("مسؤول") == "مسوول"
    assert normalize_search("شائع") == "شايع"
    assert normalize_search("Passport ٣") == "passport 3"


def test_tokens_strip_clitics_and_stopwords():
    assert tokens("الوثائق") == ["وثايق"]
    t = tokens("ما الوثائق المطلوبة لإصدار جواز السفر؟")
    assert "ما" not in t and "الوثائق" not in t
    assert tokens("بالرسوم") == tokens("الرسوم") == tokens("والرسوم") == ["رسوم"]
    assert tokens("للرسوم") == ["رسوم"]


def test_tokens_short_words_keep_prefix():
    # "بيت" must not become "يت"; "الم" style tokens with < 3 letters left stay intact
    assert tokens("بيت") == ["بيت"]
    assert tokens("وزن") == ["وزن"]


def test_tokens_stopwords_and_latin():
    assert tokens("في من على إلى عن") == []
    assert tokens("Passport 123") == ["passport", "123"]


def test_query_and_corpus_meet():
    q = set(tokens("شو الأوراق المطلوبة؟"))
    d = set(tokens("الأوراق المطلوبة: صورة شخصية"))
    assert {"اوراق", "مطلوبه"} <= q & d


def test_is_arabic():
    assert is_arabic("مرحبا بكم")
    assert not is_arabic("hello world")
    assert not is_arabic("")


def test_looks_reversed():
    ok = "يجب على المواطن تقديم الطلب في المركز من خلال الخدمة " * 3
    assert not looks_reversed(ok)
    rev = " ".join(w[::-1] for w in ok.split())
    assert looks_reversed(rev)


def test_repair_ligatures_and_list_numbers():
    from DB.parse import fix_list_numbers, repair_ligatures
    ref = "هذه الوثيقة لأغراض الاختبار فقط للمواطنين المنتهي صلاحيته"
    bad = "هذه الوثيقة ألغراض االختبار فقط للمواطنني املنتهي صالحيته وثيقة"
    fixed, n = repair_ligatures(bad, ref)
    assert fixed == "هذه الوثيقة لأغراض الاختبار فقط للمواطنين المنتهي صلاحيته وثيقة" and n == 5
    assert repair_ligatures("كلمة غريبة", ref) == ("كلمة غريبة", 0)       # no anagram -> untouched
    assert fix_list_numbers("- .1 جواز السفر\n  - .4 إيصال") == "1. جواز السفر\n  4. إيصال"
