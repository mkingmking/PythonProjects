import re
import time
import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
Hayat = """
Olumlu düşünen, bütün olası dünyaların en iyisinde yaşadığımızı söyler ve olumsuz düşünen, bunun gerçek olmasından korkar. James Cabell

Bu dünyaya gelmenin sadece tek bir yolu vardır, terk etmenin ise çok fazla yolu. Donald Harington

Birisinin ‘Hayat zor.’ diye yakındığını duyduğumda, her zaman, ‘Neye kıyasla?’ diye sormayı isterim. Sidney Harris

Mutsuz olmamamız gerekir. Kimsenin hayatla bir kontratı yok. DavidHeath

Hayatını kazanırken yaşamasını bilmeyen bir adam, servetini kazanmadan öncesine göre daha fakirdir. Josiah Holland

Yarının ne olacağını sormaktan vazgeç. Her gün, sana verilen bir hazinedir. Eline geçtikçe değerlendir. Horace

Hayatınızı yaşamanın en iyi yolu, sizden sonra da kalıcı olacak şeyler için harcamaktır. William James

Biz Japonlar, küçük zevklerden hoşlanırız, israftan değil. Daha fazlasını karşılayabilirse de, insanın basit bir yaşam tarzı olması kanısındayım. Massaru Ibuka

Benim sanatım ve mesleğim yaşamaktır. Montaigne

Hayat, çikolata ile dolu bir kutu gibidir. Ne çıkacağını asla bilemezsiniz. Forrest Gump Filmi

Hayat, büyük bir sürprizdir. Ölümün neden daha büyük bir sürpriz olması gerektiğini anlayamıyorum. Vladimir Nabokov
"""
Sevgi = """
Karım,  benim  ‘Seni seviyorum.’  dememi binlerce kez duydu ama, hiçbir zaman ‘Üzgünüm.’ dediğimi duymadı. Bruce Willis

Büyüklerine saygı, küçüklerine şevkat göstermeyenler, bizden değildir. Hz, Muhammed

Sevip de kaybetmek, sevmemiş olmaktan daha iyidir. Seneca

Sevgiyle düşünün, sevgiyle konuşun, sevgiyle davranın. Her ihtiyaç karşılanacaktır. James Ailen

Sevgi, insanı birliğe, bencillik ise yalnızlığa götürür. Schiller
"""
Gayret = """
Dileyin verilecektir; arayın bulacaksınız; kapıyı çalın size açılacaktır.    Hz.İsa (a.s.)

Damlayan su, mermeri, yürüyen de dağları deler. Ovidius

Hazine, eziyet çekene gözükür.    Hz. Mevlâna

Beklenen gün gelecekse, çekilen çile kutsaldır.V. Hugo

Yarınlar yorgun ve bezgin kimseler değil, rahatını terk edebilen gayretli insanlara aittir. Cicero

"""
Umut = """
Başlangıçta fazla umut ederiz ama, sonrasında yeteri kadar değil. Joesph Roux

Hayatta umutsuz durumlar yoktur,  sadece umutsuzluk besleyen insanlar vardır. Booth

Şafaktan önce her yer karanlıktır. Katherine Mansfield

Umut, gözle görülemeyeni görür, elle tutulamayanı hisseder ve imkansızı başarır. Anonim

"""
Idare = """
Çok söyleyen değil, çok iş yapan yöneticiye muhtaçsınız. Hz. Ömer (r.a.)

Sevginin kurduğu devleti, adalet devam ettirir. Farabi

Her memleketin hakettiği bir hükümeti vardır. J. Maistere

Hükümetlerin en kötüsü, suçsuzu korkutandır.

"""

Racon = """ Terazinin iki tarafında kimin durduğunun önemi yok. Önemli olan kefeyi tutan demir. - Mehmet Karahanlı

 Bu alemde gece mahkum olan gündüz hakim olamaz. - Laz Ziya

Namımızın büyüklüyü dostlarımızın büyüklüğündendir. Süleyman Çakır

Bir işe ya başlarken ezerler ya da başındayken ezerler. - Mehmet Karahanlı

Hasmın kapına gelecek kadar cesursa, sende karşısına çıkacak kadar cesur ol. - Laz Ziya

Gün bitti ay doğdu kurt burada, çakalların nerde aslanım? - Doğu Bey

Kahraman yapılmaz kahraman olunur! -  Kara

Aslan olmak isteyen inlemez! Aslansan hakkını vereceksin, cesareti yüreğine koyacaksın!

Sakın 30 yıl hukukun olmayan birine sakın deme… -Polat Alemdar

Bizim gibi adamlar iki yerde huzur bulurlar: Bir mezarda, bir de mapusta…

Kahpelik gizli yapılır gizli kalmaz… - Duran Emmi

Benim yaşayan bir düşmanım yok! - Polat Alemdar

Devlet ideolojisi olan insandan korkmaz.

Dostum olmaz, hasmım yaşamaz! – Laz Ziya

Bizim gibi adamlar ölümden kaçmaz!

Bu gireceğimiz ne ilk ne son tuzak.

O bizi görmeden önce taştık, o bizi tuttu cevher yaptı.

Ben Elifi kaybetmedim hiç kimsenin giremeyeceği yere sakladım. -Polat Alemdar

Biz belimize silahı, silahla vurmak için koyduk -Duran Emmi

Bir çocuk babasız büyür ama anasız büyüyemez. Bizim bir tane anamız var o da vatan! - Polat Alemdar

Dostun dostumdur düşmanın düşmanım -Polat Alemdar

Bu kapıya aç giren çok olur, aç çıkan hiç olmaz.

Geçmişini unutan geleceğini bulamaz -Polat Alemdar

Duygu güçsüzlerin sığınağıdır, onların mabedidir.

Ben buraya kan dökmeye değil kanı durdurmaya geldim -Polat Alemdar

Sen kimsin ki benim vermediğim şansı veriyorsun.

İtaatsizliğin raconunu biz koymadık ama biz uygularız -Polat Alemdar

Bir insanın yanında dişisi varken ne onunla konuş ne de ona bir laf anlat.

Haddini bilmek az sonra ölecekmiş gibi yaşamaktır -Polat Alemdar

Gün bitti ay doğdu kurt burada, çakalların nerede aslanım?

Bu kurumu da bu koltuğu da sana yedirtmem Vural keyfini sür. Ben seni oturtacağım yeri biliyorum -Polat Alemdar

Kurtlar Vadisi’nde bela, kişinin sevdiklerinden gelir.

Yürü İskenderim yürü verecek hesabın çok anca gideriz -Polat Alemdar

Sana kim dedi ki şeref karın doyuruyor?

Sen hiç mezar taşına delikanlı yazıldığını gördün mü?

Artık inat edilecek zaman değil, akılla hareket edilecek zaman.

Ben bitti demeden bitmez her şey yeni başlıyor -Polat Alemdar

Kurtlar Vadisi’nde sonunu düşünenler kahraman olamaz.

Hainle kahraman arasındaki farkı mahkemeler değil tarih belirler -Polat Alemdar

Ben verebilecek olandan bir şey isterim, olmayandan değil. -Polat Alemdar

Azdan az Çoktan çok gider... -Süleyman Çakır"""






categories = [Hayat,Sevgi,Gayret,Umut,Idare,Racon]
categories_name = ["hayat", "sevgi", "gayret", "umut", "idare", "racon"]
for i in range(len(categories)):
    categories[i] = categories[i].split("\n")
df = pd.DataFrame(columns=['sentence','category'])
for i in range(len(categories_name)):
    for k in range(len(categories[i])):
        df = df.append({'sentence': categories[i][k], 'category': categories_name[i]}, ignore_index=True)
    
    df = df[df['sentence']!=""]
    df.reset_index(drop=True, inplace=True)






df["cleaned_sentence"]=df['sentence'].copy()

for i in range(len(df)):
    df["cleaned_sentence"][i] = re.sub('[!@#’‘?.,\'$]', '', df["cleaned_sentence"][i])
    df["cleaned_sentence"][i] = df["cleaned_sentence"][i].lower()



def rp_string(string: str):
    string_entered = ""
    for j in range(len(string)):
        if string[j].isalpha() or string[j].isdigit() or string[j].isspace():
          string_entered += string[j]
    return string_entered




start = time.time()

word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)

print("total time: ", time.time() - start)


class life_coach_w2v:
    def construct_wv_matrices(df, answer):
        word_matrix = np.zeros((len(df), 400))
        for i in range(len(df)):
            wv_sum = np.zeros((1, 400))
            for j in df.iloc[i]['cleaned_sentence'].split(' '):
                try:
                    wv_sum += word_vectors[j]
                except: 
                    pass
            word_matrix[i] = np.divide(wv_sum, len(df.iloc[i]['cleaned_sentence'].split(' ')))

            answer_vector = np.zeros((1, 400))
            for k in answer.split(' '):
                try:
                    answer_vector += word_vectors[k]
                except: 
                    pass
            answer_vector = np.divide(answer_vector, len(answer.split(' ')))

        return word_matrix, answer_vector

    def recommendation_wv(answer):
        
        answer = answer.lower()

        word_matrix, answer_vector = life_coach_w2v.construct_wv_matrices(df, answer)

        score_board = []
        for i in range(len(df)):
            score = cosine_similarity(word_matrix[i].reshape(1,400), answer_vector.reshape(1,400))
            score_board.append(score)
        recommendation = df.iloc[score_board.index(max(score_board))].sentence
        if("hayat" == df.iloc[score_board.index(max(score_board))].category):
            st.image("hayat.jpg")
        if("sevgi" == df.iloc[score_board.index(max(score_board))].category):
            st.image("sevgi.jpg")
        if("gayret" == df.iloc[score_board.index(max(score_board))].category):
            st.image("gayret.jpg")
        if("umut" == df.iloc[score_board.index(max(score_board))].category):
            st.image("umut.jpg")
        if("idare" == df.iloc[score_board.index(max(score_board))].category):
            st.image("idare.jpg")
        if("racon" == df.iloc[score_board.index(max(score_board))].category):
            st.image("racon.jpg")

        st.write(recommendation)

        return

        
        
        return recommendation
st.title("Yaşam Koçum Uygulaması")
satir = st.text_input("Tavsiye için cümle giriniz! (Yeni Racon kategorimizi deneyebilirsiniz)")
if st.button('Enter'):
    life_coach_w2v.recommendation_wv(rp_string(satir).lower())   



