from pyknp import Juman
from pyknp import KNP
def jumancategory(token):
    jumanpp = Juman()
    result = jumanpp.analysis(token)
    category = ""
    for mrph in result.mrph_list():
        print("見出し:%s, 読み:%s, 原形:%s, 品詞:%s, 品詞細分類:%s, 活用型:%s, 活用形:%s, 意味情報:%s, 代表表記:%s" \
                % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))
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
    #[knpparsedic["".join(mrph.midasi for mrph in bnst_dic[bnst.parent_id].mrph_list())].append(list("".join(mrph.midasi for mrph in bnst.mrph_list()))) if ("".join(mrph.midasi for mrph in bnst_dic[bnst.parent_id].mrph_list())) in knpparsedic.keys() else knpparsedic[("".join(mrph.midasi for mrph in bnst_dic[bnst.parent_id].mrph_list()))] = list("".join(mrph.midasi for mrph in bnst.mrph_list())) ]
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