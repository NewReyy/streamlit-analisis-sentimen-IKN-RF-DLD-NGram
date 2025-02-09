# =======================
# 1. LOAD MODEL & VECTOR
# =======================
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import re
import textdistance
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

# =======================
# 2. LOAD PRE-TRAINED MODELS
# =======================
# Load dataframe hasil preprocessing
with open("cleaned_data.pkl", "rb") as file:
    cleaned_df = pickle.load(file)

with open("casefolded_data.pkl", "rb") as file:
    casefolded_df = pickle.load(file)

with open("tokenized_data.pkl", "rb") as file:
    tokenized_df = pickle.load(file)

with open("normalized_data.pkl", "rb") as file:
    normalized_df = pickle.load(file)

with open("filtered_data.pkl", "rb") as file:
    filtered_df = pickle.load(file)

with open("stemmed_data.pkl", "rb") as file:
    stemmed_df = pickle.load(file)
    
with open("tfidf_data_unigram.pkl", "rb") as file:
    tfidf_df_unigram = pickle.load(file)

with open("tfidf_data_bigram.pkl", "rb") as file:
    tfidf_df_bigram = pickle.load(file)

@st.cache_resource
def load_model(model_type):
    if model_type == "Random Forest Tanpa DLD & N-Gram":
        model_file = "random_forest_model_0,0006_Bigram.pkl"
        vectorizer_file = "vectorizer_bigram_new.pkl"
    else:
        model_file = "random_forest_model_0,0002.pkl"
        vectorizer_file = "vectorizer-new.pkl"

    with open(model_file, 'rb') as model_f:
        model = pickle.load(model_f)
    with open(vectorizer_file, 'rb') as vectorizer_f:
        vectorizer = pickle.load(vectorizer_f)

    return model, vectorizer

# =======================
# 3. PREPROCESSING FUNCTION
# =======================
import re
import string

def cleaningText(text):
    # menghilangkan url
    text = re.sub(r'https?:\/\/\S+','',text)
    # menghilangkan mention, link, hastag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #menghilangkan karakter byte (b')
    text = re.sub(r'(b\'{1,2})',"", text)
    # menghilangkan yang bukan huruf
    text = re.sub('[^a-zA-Z]', ' ', text)
    # menghilangkan digit angka
    text = re.sub(r'\d+', '', text)
    #menghilangkan tanda baca
    text = text.translate(str.maketrans("","",string.punctuation))
    # menghilangkan whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Memecah teks menjadi kata-kata terpisah
    text = text.split()

    return ' '.join(text)  # Mengembalikan teks yang sudah diproses kembali menjadi string

def casefoldingText(text):
    text = text.lower()
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text

nltk.download('punkt_tab')

kamus_normalisasi = pd.read_csv("slang.csv")

kata_normalisasi_dict = {}

for index, row in kamus_normalisasi.iterrows():
    if row[0] not in kata_normalisasi_dict:
        kata_normalisasi_dict[row[0]] = row[1]

def normalisasi_kata(document):
    return [kata_normalisasi_dict[term] if term in kata_normalisasi_dict else term for term in document]

import re

def filteringText(text):  # Remove stopwords in a text
    listStopwords = stopwords.words('indonesian')

    filtered = []
    for txt in text:
        if txt not in listStopwords:  # Cek stopwords dan validitas kata
            filtered.append(txt)

    return filtered

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmingText(token_list):
    # Daftar pengecualian
    exception_list = {
            "pengen": "pengen",
            "makan": "makan",
            "minum": "minum",
            "bermain": "main",
            "mengapa": "mengapa",
            "ingin": "ingin",
            "akan": "akan",
            "dengan": "dengan",
            "tidak": "tidak"
        }

    # stopword
    with open('kamus.txt') as kamus:
        word = kamus.readlines()
        list_stopword = [line.replace('\n', "") for line in word]
    dictionary = ArrayDictionary(list_stopword)
    stopword = StopWordRemover(dictionary)

    # Inisialisasi list untuk menyimpan hasil stemming
    stemmed_token_list = []

    # Proses setiap token dalam list
    for token in token_list:
        # Cek apakah token ada dalam daftar pengecualian
        if token in exception_list:
            stemmed_token_list.append(exception_list[token])
        else:
            # Hapus stopword
            token = stopword.remove(token)

            # Jika token tidak termasuk dalam stopword, lakukan stemming
            if token:
                stemmed_token = stemmer.stem(token)
                stemmed_token_list.append(stemmed_token)

    return stemmed_token_list

# =======================
# 4. SPELL CHECKER FUNCTION
# =======================
dictionary = set(pd.read_csv("kamusDLD.csv")['Unique_Words'].dropna().str.lower())

def spell_checker(tokens):
    corrected_tokens = []
    for word in tokens:
        if word in dictionary:
            corrected_tokens.append(word)
        else:
            suggestion = min(dictionary, key=lambda x: textdistance.damerau_levenshtein(word, x))
            corrected_tokens.append(suggestion)
    return corrected_tokens

# =======================
# 5. PREDICTION FUNCTION
# =======================

def predict_sentiment(text):
    if 'vectorizer' in st.session_state and 'model' in st.session_state:
        vectorizer = st.session_state.vectorizer
        model = st.session_state.model
        
        print("Apakah vectorizer sudah di-fit?", hasattr(vectorizer, "idf_"))
        
        # Jika menggunakan vectorizer bigram tetapi teks kurang dari 2 kata
        if hasattr(vectorizer, 'ngram_range') and vectorizer.ngram_range == (1, 2):
            if len(text.split()) < 2:
                return "Teks terlalu pendek untuk bigram, dilewati."

        text_vectorized = vectorizer.transform([text])  # Transformasi teks

        # Pastikan jumlah fitur tetap sama
        expected_features = model.n_features_in_
        current_features = text_vectorized.shape[1]

        if current_features != expected_features:
            # Konversi ke bentuk DataFrame
            text_vectorized = pd.DataFrame.sparse.from_spmatrix(text_vectorized, columns=vectorizer.get_feature_names_out())

            # Tambahkan fitur yang hilang
            missing_features = set(model.feature_names_in_) - set(text_vectorized.columns)
            for feature in missing_features:
                text_vectorized[feature] = 0  # Isi dengan nol

            # Urutkan fitur agar sesuai dengan model
            text_vectorized = text_vectorized[model.feature_names_in_]

        # Prediksi
        prediction = model.predict(text_vectorized)[0]
        return "Positif" if prediction == 1 else "Negatif"
    
    else:
        return "Model belum dipilih!"

# =======================
# 6. STREAMLIT DASHBOARD
# =======================
st.set_page_config(page_title="Dashboard Analisis Sentimen", layout="wide")

# Sidebar Navigasi
with st.sidebar:
    menu = option_menu(
        menu_title="Analisis Sentimen",
        options=["Home", "Preprocessing", "Pemilihan Model", "Analisis Sentimen", "Model Performance", "Tentang Aplikasi"]
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 18px;">
            © 2025 Nuri Hidayatuloh. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )
# =======================
# 7. HOME
# =======================
if menu == "Home":
    st.title("Dashboard Analisis Sentimen")
    st.write("Powered by Nuri Hidayatuloh")
    st.subheader("Analisis Sentimen Relokasi Ibu Kota Nusantara Dengan Random Forest Menggunakan Damerau-Levenshtein Distance dan N-Gram")
    st.write("Penelitian ini menganalisis sentimen terkait relokasi Ibu Kota Nusantara (IKN) menggunakan Random Forest sebagai model klasifikasi. " 
             "Damerau-Levenshtein Distance digunakan untuk pemeriksaan ejaan, sedangkan N-Gram digunakan dalam pemrosesan teks untuk memahami pola kata. "
             "Pendekatan ini bertujuan meningkatkan akurasi analisis sentimen dan mengkaji opini publik tentang pemindahan IKN.")
    st.write("Akurasi Klasifikasi dalam penelitian ini mencapai sampai dengan 90,21% dengan akurasi terbaik")

# =======================   
# 8. PEMILIHAN MODEL
# =======================
elif menu == "Pemilihan Model":
    st.title("Pilih Model Pre-trained")
    model_option = st.selectbox("Pilih Model:", ["Random Forest Tanpa DLD & N-Gram", "Random Forest Menggunakan DLD & N-Gram"])

    if st.button("Load Model"):
        model, vectorizer = load_model(model_option)
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.success("Model berhasil dimuat!")

# =======================
# 9. ANALISIS SENTIMEN
# =======================
elif menu == "Analisis Sentimen":
    st.header("Analisis Sentimen")

    tab1, tab2 = st.tabs(["Input Teks Manual", "Analisis Batch (Upload CSV)"])

    with tab1:
        st.subheader("Input Teks Manual")
        user_input = st.text_area("Masukkan teks di sini:")

        if st.button("Periksa Ejaan"):
            if user_input.strip():
                cleaned_text = cleaningText(user_input)
                folded_text = casefoldingText(cleaned_text)
                tokens = tokenizingText(folded_text)
                normalized_tokens = normalisasi_kata(tokens)
                filtered_tokens = filteringText(normalized_tokens)
                stemmed_tokens = stemmingText(filtered_tokens)
                corrected_tokens = spell_checker(stemmed_tokens)
                
                df = pd.DataFrame({
                    "Tahap": ["Original", "Cleaning", "Case Folding", "Tokenizing", "Normalization", "Filtering", "Stemming", "Spell Correction"],
                    "Hasil": [user_input, cleaned_text, folded_text, tokens, normalized_tokens, filtered_tokens, stemmed_tokens, corrected_tokens]
                })
                st.write("### Hasil Preprocessing & Spell Checker:")
                st.write(df)
                
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")

        if st.button("Analisis Sentimen"):
            if user_input.strip():
                # **Preprocessing**
                cleaned_text = cleaningText(user_input)
                folded_text = casefoldingText(cleaned_text)
                tokens = tokenizingText(folded_text)
                normalized_tokens = normalisasi_kata(tokens)
                filtered_tokens = filteringText(normalized_tokens)
                stemmed_tokens = stemmingText(filtered_tokens)
                corrected_tokens = spell_checker(stemmed_tokens)
                
                # **Prediksi Sentimen**
                sentiment = predict_sentiment(" ".join(corrected_tokens))

                # **Tampilkan Hasil Sentimen**
                st.info(f"**Hasil Sentimen:** {sentiment}")
                df1 = pd.DataFrame({
                    "Tahap": ["Original", "Cleaning", "Case Folding", "Tokenizing", "Normalization", "Filtering", "Stemming", "Spell Correction", "Sentimen"],
                    "Hasil": [user_input, cleaned_text, folded_text, tokens, normalized_tokens, filtered_tokens, stemmed_tokens, corrected_tokens, sentiment]
                })
                st.write("### Hasil Sentimen Analisis:")
                st.write(df1)
            else:
                    st.warning("Silakan masukkan teks terlebih dahulu.")

    # --- Tab 2: Analisis Batch (Upload CSV) ---
    with tab2:
        st.subheader("Analisis Batch (Upload CSV)")
        uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                # Mencoba membaca CSV dengan berbagai kemungkinan delimiter
                df = pd.read_csv(uploaded_file, sep='[,;]+', engine='python', encoding='utf-8', on_bad_lines='skip')
                st.write("File CSV berhasil dimuat!")
                st.dataframe(df.head())  # Menampilkan beberapa baris pertama
                
            except pd.errors.ParserError:
                st.error("Format file CSV tidak valid. Pastikan file memiliki delimiter yang benar (`,` atau `;`).")
            except UnicodeDecodeError:
                st.error("Encoding file tidak didukung. Coba simpan ulang file CSV dalam format UTF-8.")

            if 'teks' not in df.columns:
                st.warning("Pastikan ada kolom 'teks' di file CSV Anda.")
            else:
                if st.button("Periksa Ejaan (Batch)"):
                    df['teks_cleaning'] = df['teks'].apply(cleaningText)
                    st.write("### Hasil Cleaning:")
                    st.dataframe(df[['teks', 'teks_cleaning']].head())
                    
                    df['teks_casefolding'] = df['teks_cleaning'].apply(casefoldingText)
                    st.write("### Hasil Case Folding:")
                    st.dataframe(df[['teks_cleaning', 'teks_casefolding']].head())
                    
                    df['teks_tokenizing'] = df['teks_casefolding'].apply(tokenizingText)
                    st.write("### Hasil Tokenizing:")
                    st.dataframe(df[['teks_casefolding', 'teks_tokenizing']].head())
                    
                    df['teks_normalisasi'] = df['teks_tokenizing'].apply(normalisasi_kata)
                    st.write("### Hasil Normalisasi:")
                    st.dataframe(df[['teks_tokenizing', 'teks_normalisasi']].head())
                    
                    df['teks_filtering'] = df['teks_normalisasi'].apply(filteringText)
                    st.write("### Hasil Filtering:")
                    st.dataframe(df[['teks_normalisasi', 'teks_filtering']].head())
                    
                    df['teks_stemming'] = df['teks_filtering'].apply(stemmingText)
                    st.write("### Hasil Stemming:")
                    st.dataframe(df[['teks_normalisasi', 'teks_stemming']].head())
                    
                    df['teks_DLD'] = df['teks_stemming'].apply(spell_checker)
                    st.write("### Hasil Preprocessing & Spell Checking:")
                    st.dataframe(df[['teks_stemming', 'teks_DLD']].head())

                if st.button("Analisis Sentimen (Batch)"):
                    df['teks_cleaning'] = df['teks'].apply(cleaningText)
                    st.write("### Hasil Cleaning:")
                    st.dataframe(df[['teks', 'teks_cleaning']].head())
                    
                    df['teks_casefolding'] = df['teks_cleaning'].apply(casefoldingText)
                    st.write("### Hasil Case Folding:")
                    st.dataframe(df[['teks_cleaning', 'teks_casefolding']].head())
                    
                    df['teks_tokenizing'] = df['teks_casefolding'].apply(tokenizingText)
                    st.write("### Hasil Tokenizing:")
                    st.dataframe(df[['teks_casefolding', 'teks_tokenizing']].head())
                    
                    df['teks_normalisasi'] = df['teks_tokenizing'].apply(normalisasi_kata)
                    st.write("### Hasil Normalisasi:")
                    st.dataframe(df[['teks_tokenizing', 'teks_normalisasi']].head())
                    
                    df['teks_filtering'] = df['teks_normalisasi'].apply(filteringText)
                    st.write("### Hasil Filtering:")
                    st.dataframe(df[['teks_normalisasi', 'teks_filtering']].head())
                    
                    df['teks_stemming'] = df['teks_filtering'].apply(stemmingText)
                    st.write("### Hasil Stemming:")
                    st.dataframe(df[['teks_normalisasi', 'teks_stemming']])
                    
                    df.drop_duplicates(subset = 'teks_stemming', inplace = True)
                    df.dropna(inplace = True)
                    df.reset_index(drop = True, inplace = True)
                    
                    df['teks_DLD'] = df['teks_stemming'].apply(spell_checker)
                    st.write("### Hasil Preprocessing & Spell Checking:")
                    st.dataframe(df[['teks_stemming', 'teks_DLD']].head())
                    
                    # Cek apakah model yang digunakan adalah bigram
                    is_bigram = False
                    if 'vectorizer' in st.session_state:
                        vectorizer = st.session_state.vectorizer
                        if hasattr(vectorizer, 'ngram_range') and vectorizer.ngram_range == (1, 2):
                            is_bigram = True

                    # Lakukan preprocessing untuk batch
                    df['teks_DLD'] = df['teks_DLD'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

                    # Hapus teks yang kosong
                    df = df[df['teks_DLD'].str.strip() != ""]

                    # Jika menggunakan bigram, hapus teks yang kurang dari 2 kata
                    if is_bigram:
                        df = df[df['teks_DLD'].str.count(" ") >= 2]  # Minimal 2 kata agar bisa menjadi bigram

                    # Lakukan prediksi sentimen
                    df['sentimen'] = df['teks_DLD'].apply(lambda x: predict_sentiment(x))

                    # Tampilkan hasil
                    st.write("### Hasil Analisis Sentimen:")
                    st.dataframe(df[['teks_stemming', 'teks_DLD', 'sentimen']])

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Unduh Hasil", data=csv, file_name="hasil_analisis_sentimen.csv", mime="text/csv")

# =======================
# 10. Model Performance
# =======================
elif menu == "Model Performance":
    st.header("Model Performance")
    
    # Data hasil prediksi berdasarkan confusion matrix yang diberikan
    data_model = {
        "Random Forest Tanpa DLD & N-Gram Terbaik": {
            "y_true": np.array([0] * 100 + [1] * 94),
            "y_pred": np.array([0] * 91 + [1] * 9 + [0] * 11 + [1] * 83)
        },
        "Random Forest Dengan DLD & N-Gram Terbaik": {
            "y_true": np.array([0] * 100 + [1] * 94),
            "y_pred": np.array([0] * 90 + [1] * 10 + [0] * 9 + [1] * 85)
        }
    }

    # Header Streamlit
    st.header("Performa Kinerja Model")
    st.write("Pilih model yang ingin dilihat performanya:")

    # Dropdown pilihan model
    selected_model = st.selectbox("Pilih Model", list(data_model.keys()))

    # Tombol submit
    if st.button("Submit"):
        data = data_model[selected_model]
        y_true, y_pred = data["y_true"], data["y_pred"]
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Layout 2 kolom
        col1, col2 = st.columns(2)

        # Kolom 1 - Confusion Matrix
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))  # Ukuran lebih kecil
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 8}, cbar=False)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.set_xticklabels(["Negatif", "Positif"], fontsize=7)
            ax.set_yticklabels(["Negatif", "Positif"], fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

        # Kolom 2 - Tabel Metrik
        with col2:
            st.subheader("Metrics")
            accuracy = accuracy_score(y_true, y_pred) * 100
            precision = precision_score(y_true, y_pred, zero_division=1) * 100
            recall = recall_score(y_true, y_pred, zero_division=1) * 100
            f1 = f1_score(y_true, y_pred, zero_division=1) * 100

            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Value": [f"{accuracy:.2f}%", f"{precision:.2f}%", f"{recall:.2f}%", f"{f1:.2f}%"]
            })

            # Gunakan st.dataframe agar tabel lebih rapi
            st.dataframe(metrics_df, height=175, width=200)

# =======================
# 10. Preprocessing
# =======================
elif menu == "Preprocessing":
    st.header("Preprocessing")
    
    def display_limited_rows(df, title):
        st.subheader(title)
        st.dataframe(pd.concat([df.head(10)]))

    display_limited_rows(cleaned_df, "Hasil Cleaning")
    display_limited_rows(casefolded_df, "Hasil Case Folding")
    display_limited_rows(tokenized_df, "Hasil Tokenizing")
    display_limited_rows(normalized_df, "Hasil Normalization")
    display_limited_rows(filtered_df, "Hasil Filtering")
    display_limited_rows(stemmed_df, "Hasil Stemming")
    display_limited_rows(tfidf_df_unigram, "Hasil TF-IDF Tanpa N-Gram")
    display_limited_rows(tfidf_df_bigram, "Hasil TF-IDF Bigram")

# =======================
# 11. TENTANG APLIKASI
# =======================
elif menu == "Tentang Aplikasi":
    st.header("Tentang Aplikasi")
    
    st.write("""
    Aplikasi ini merupakan **Dashboard Analisis Sentimen** yang dikembangkan sebagai bagian dari penelitian skripsi 
    Analisis Sentimen Relokasi Ibu Kota Nusantara. Aplikasi ini memungkinkan pengguna untuk 
    menganalisis sentimen terhadap berbagai isu terkait dengan IKN dengan pendekatan berbasis machine learning.
    """)

    st.subheader("🎯 Fitur Utama")
    st.markdown("""
    - 🔍 **Analisis sentimen** menggunakan **Random Forest Classifier**  
    - 📝 **Pemeriksa ejaan** berbasis **Damerau-Levenshtein Distance**  
    - 📂 **Dukungan input manual** dan **analisis batch (upload CSV)**  
    - 📊 **Visualisasi data interaktif** untuk memahami pola sentimen  
    """)

    st.subheader("🎯 Tujuan Pengembangan")
    st.markdown("""
    - Memahami persepsi masyarakat terhadap Ibu Kota Nusantara (IKN) dan isu sosial lainnya  
    - Memberikan wawasan berbasis data untuk mendukung pengambilan keputusan  
    - Menyediakan alat analisis yang mudah digunakan bagi peneliti dan pengambil kebijakan  
    """)

    st.subheader("🛠 Teknologi yang Digunakan")
    st.markdown("""
    - **Streamlit** untuk pembuatan antarmuka dashboard  
    - **Scikit-learn** untuk model Machine Learning  
    - **NLTK / Sastrawi** untuk pemrosesan teks  
    - **Damerau-Levenshtein Distance** untuk pemeriksa ejaan  
    """)
    
    st.subheader("👤 Profil Author")
    st.write("Aplikasi ini dikembangkan oleh **[Nuri Hidayatuloh]**, seorang mahasiswa yang tertarik dengan analisis data, data engineer, machine learning, dan pengolahan teks.")

    st.markdown("""
    🔗 **Follow Me in Sosial Media**  
    - [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/nuri-hidayatuloh)  
    - [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/NewReyy)  
    - [![Instagram](https://img.shields.io/badge/Instagram-Profile-purple?logo=instagram)](https://instagram.com/nr.hdytlh)  
    - [![Facebook](https://img.shields.io/badge/Facebook-Profile-blue?logo=facebook)](https://facebook.com/nuree.hidayatuloh)  
    """)