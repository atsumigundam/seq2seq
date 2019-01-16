from pyknp import Juman
from pyknp import KNP
def jumancategory(token):
    jumanpp = Juman()
    result = jumanpp.analysis(token)
    category = ""
    for mrph in result.mrph_list():
        if "カテゴリ:" in mrph.imis:
            category = mrph.imis.split('カテゴリ:')[1].split(";")[0]
        elif "人名" in mrph.imis:
            category = "人"
    return category
def pyknpbnstparser(line):
    knp = KNP()
    result = knp.parse(line)
    bnst_list = result.bnst_list()
    bnst_dic = dict((x.bnst_id, x) for x in bnst_list)
    knpparsedic = {}
    for bnst in bnst_list:
        if bnst.parent_id != -1:
            advancebnst = "".join(mrph.midasi for mrph in bnst_dic[bnst.parent_id].mrph_list())
            correntbnst = "".join(mrph.midasi for mrph in bnst.mrph_list())
            if advancebnst not in knpparsedic.keys():
                knpparsedic[advancebnst] = [correntbnst]
            else:
                knpparsedic[advancebnst].append(correntbnst)
    return knpparsedic

if __name__ == '__main__':
    line = "太郎は花子が読んでいる本を次郎に渡した"
    token = "太郎"
    category =jumancategory(token)
    knpbnst = pyknpbnstparser(line)
    print(category)
    print(knpbnst)