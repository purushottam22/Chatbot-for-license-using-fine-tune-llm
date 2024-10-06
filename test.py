# pip install ai4bharat-transliteration
# pip install editdistance
# pip install wheel
# pip install fairseq
from ai4bharat.transliteration import XlitEngine


lang_dic = {'Assamese' , 'Bengali', 'Gujarati', 'Hindi', 'Kannada', 'Kashmiri', 'Malayalam', 'Marathi', 'Oriya',
            'Punjabi', 'Sanskrit', 'Tamil', 'Telugu',  'Urdu'}
e = XlitEngine("hi", beam_width=10, rescore=True)
out = e.translit_word("namasthe", topk=5)