"""api.best_line: the viewer highlights the line(s) the answer cites, not the whole chunk / website page."""
import api

PAGE = """الرئيسية
الخدمات
دليل الخدمات
( بطاقة الأحوال المدنية الذكية بدل فاقد)
الوثائق المطلوبة للحصول على الخدمة 1. نموذج البطاقة الشخصية
2. اثبات شخصية لصاحب العلاقة
3.التعهد الخطي اذا كان الفقدان للمرة الاولى او الثانية
4.محضر افادة بالفقدان من لجنة التحقيق اذا كان الفقدان للمرة الثالثة
9. بطاقة إحصاءات الجسور لمن يلزمهم ذلك( الصورة+ الأصل) أو ختم المتابعة والتفتيش.
الزمن المستغرق لتقديم الخدمة 20 دقيقة
الرسوم 5 دنانير"""


def hl(answer, n=1):
    a, b = api.best_line(PAGE, 0, len(PAGE), answer, n)
    return PAGE[a:b]


def test_single_cited_fact_is_its_line():
    assert hl("الرسوم 5 دنانير [1].") == "الرسوم 5 دنانير"
    assert hl("تستغرق الخدمة 20 دقيقة [1].") == "الزمن المستغرق لتقديم الخدمة 20 دقيقة"


def test_cited_list_covers_its_lines_not_the_website_menu():
    out = hl("1. نموذج البطاقة الشخصية [1]\n2. التعهد الخطي اذا كان الفقدان للمرة الأولى [1]\n"
             "3. محضر افادة بالفقدان من لجنة التحقيق [1]")
    assert out.startswith("الوثائق المطلوبة") and out.endswith("للمرة الثالثة") and "الرئيسية" not in out


def test_list_numbers_do_not_pull_the_highlight():
    assert "الجسور" not in hl("1. نموذج البطاقة الشخصية [1]")


def test_citation_alone_on_a_line_uses_the_whole_answer():
    assert hl("الرسوم 5 دنانير.\n[1]") == "الرسوم 5 دنانير"


def test_one_citation_at_the_end_of_a_list_answer():
    out = hl("الوثائق المطلوبة هي:\n1. نموذج البطاقة الشخصية.\n2. التعهد الخطي اذا كان الفقدان للمرة الأولى.\n"
             "3. محضر افادة بالفقدان من لجنة التحقيق. [1]")
    assert out.startswith("الوثائق المطلوبة") and "الرسوم" not in out and "الزمن" not in out


def test_citation_only_on_a_remark_about_the_document():
    out = hl("الوثائق المطلوبة:\n1. نموذج البطاقة الشخصية.\n2. محضر افادة بالفقدان من لجنة التحقيق.\n"
             "هذه الوثائق مذكورة في وثيقة (id_card_lost.txt) [1].")
    assert out.startswith("الوثائق المطلوبة") and out.endswith("للمرة الثالثة")


def test_items_ending_with_hasab_al_darura_are_facts_not_remarks():
    page = ("الوثائق المطلوبة للحصول على الخدمة 1. جواز السفر التالف\n2. البطاقة الشخصية لمتلقي الخدمة\n"
            "3. دفتر خدمة العلم أو شهادة إنهاء الخدمة للذكور\n4. حجة الوصاية لمن يلزمهم ذلك لغايات إصدار جواز لقاصر\n"
            "5. كفالة مالية او تعهد شخصي بالمحافظة على الجواز\n6. صور شخصية ملونة عدد 2.")
    answer = ("1. جواز السفر التالف.\n2. البطاقة الشخصية لمتلقي الخدمة.\n3. دفتر خدمة العلم أو شهادة إنهاء الخدمة "
              "للذكور (حسب الضرورة).\n4. حجة الوصاية لقاصر (حسب الضرورة).\n5. كفالة مالية أو تعهد شخصي.\n"
              "6. صور شخصية ملونة عدد 2.\nهذه الوثائق حسب الوثيقة [1].")
    a, b = api.best_line(page, 0, len(page), answer, 1)
    assert page[a:b].startswith("الوثائق المطلوبة") and page[a:b].endswith("عدد 2.")


def test_nothing_matches_keeps_a_short_span_but_never_a_whole_page():
    assert api.best_line(PAGE, 10, 40, "شيء آخر تماماً [1]", 1) == (10, 40)
    long_page = PAGE + "\n" + "سطر إضافي لا علاقة له بالسؤال\n" * 30
    assert api.best_line(long_page, 0, len(long_page), "شيء آخر تماماً [1]", 1) == (0, 0)


def test_list_beats_title_when_the_answer_heading_repeats_it():
    page = ("( تجديد بطاقة الأحوال المدنية الذكية)\n" + "سطر من قائمة الموقع لا علاقة له\n" * 8 +
            "الوثائق المطلوبة 1. نموذج البطاقة الشخصية\n2. البطاقة الشخصية المراد تجديدها\n"
            "4. بطاقة إحصاءات الجسور أو ختم المتابعة والتفتيش.\n" + "سطر آخر\n" * 10 + "الرسوم 2 دينار")
    answer = ("شروط تجديد بطاقة الأحوال المدنية الذكية هي:\n1. تقديم نموذج البطاقة الشخصية.\n"
              "2. تقديم البطاقة الشخصية المراد تجديدها.\n3. بطاقة إحصاءات الجسور أو ختم المتابعة والتفتيش.\n"
              'هذه الشروط موجودة في وثيقة "تجديد بطاقة الأحوال المدنية الذكية" [1].')
    a, b = api.best_line(page, 0, len(page), answer, 1)
    assert page[a:b].startswith("الوثائق المطلوبة") and page[a:b].endswith("والتفتيش.")
